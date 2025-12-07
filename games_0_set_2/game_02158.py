
# Generated: 2025-08-27T19:26:30.633105
# Source Brief: brief_02158.md
# Brief Index: 2158

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to slice."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit to score points while avoiding bombs in this fast-paced arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 10000
    WIN_SCORE = 100
    MAX_LIVES = 3
    CURSOR_SPEED = 20
    
    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (30, 60, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_SLICE_TRAIL = (255, 255, 240)
    
    FRUIT_TYPES = {
        "apple": {"color": (220, 30, 30), "radius": 15},
        "orange": {"color": (255, 150, 20), "radius": 16},
        "banana": {"color": (255, 220, 50), "radius": 18}, # Represented as a circle for collision
    }
    BOMB_COLOR = (20, 20, 20)
    BOMB_RADIUS = 18
    BOMB_FUSE_COLOR = (255, 50, 50)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # RNG
        self._np_random = None
        
        # Initialize state variables
        self.cursor_pos = pygame.Vector2(0, 0)
        self.last_cursor_pos = pygame.Vector2(0, 0)
        self.was_space_held = False
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.slice_trails = []
        self.fall_speed = 0.0
        self.spawn_prob = 0.0

        self.reset()
        
        # This check is a final verification
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed=seed)
        else:
            self._np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        
        self.cursor_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.last_cursor_pos = self.cursor_pos.copy()
        self.was_space_held = False
        
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.slice_trails = []
        
        self.fall_speed = 2.0
        self.spawn_prob = 0.02
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Handle Action ---
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is not used in this game

        self.last_cursor_pos = self.cursor_pos.copy()

        if movement == 1: # Up
            self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: # Down
            self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: # Left
            self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: # Right
            self.cursor_pos.x += self.CURSOR_SPEED
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        # Detect slice on space press (rising edge)
        is_slicing = space_held and not self.was_space_held
        if is_slicing:
            self.slice_trails.append({"start": self.last_cursor_pos, "end": self.cursor_pos, "life": 5})
            # sfx: whoosh_sound()

        # --- 2. Update Game State ---
        
        # Update and filter slice trails
        self.slice_trails = [t for t in self.slice_trails if t["life"] > 0]
        for trail in self.slice_trails:
            trail["life"] -= 1
            
        # Update and filter particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"].y += 0.1 # Gravity
            p["life"] -= 1

        # Spawn new items
        if self._np_random.random() < self.spawn_prob:
            self._spawn_item()

        # Update fruits
        surviving_fruits = []
        for fruit in self.fruits:
            fruit["pos"].y += self.fall_speed
            sliced = False
            if is_slicing:
                if self._check_slice_collision(fruit["pos"], fruit["radius"]):
                    self._create_fruit_particles(fruit)
                    reward += 5
                    self.score += 1
                    sliced = True
                    # sfx: fruit_slice_sound()
            
            if not sliced:
                if fruit["pos"].y > self.HEIGHT + fruit["radius"]:
                    reward -= 1 # Penalty for missing
                else:
                    surviving_fruits.append(fruit)
        self.fruits = surviving_fruits

        # Update bombs
        surviving_bombs = []
        for bomb in self.bombs:
            bomb["pos"].y += self.fall_speed
            hit = False
            if is_slicing:
                if self._check_slice_collision(bomb["pos"], self.BOMB_RADIUS):
                    self._create_bomb_particles(bomb)
                    reward -= 20
                    self.lives -= 1
                    hit = True
                    # sfx: bomb_explosion_sound()

            if not hit:
                if bomb["pos"].y > self.HEIGHT + self.BOMB_RADIUS:
                    pass # No penalty for avoiding bombs that go off-screen
                else:
                    surviving_bombs.append(bomb)
        self.bombs = surviving_bombs

        # --- 3. Update Difficulty ---
        if self.steps % 500 == 0 and self.steps > 0:
            self.fall_speed = min(6.0, self.fall_speed + 0.2)
            self.spawn_prob = min(0.1, self.spawn_prob + 0.005)

        # --- 4. Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
        if self.lives <= 0:
            reward -= 100
            terminated = True
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.was_space_held = space_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_item(self):
        x = self._np_random.integers(low=50, high=self.WIDTH - 50)
        pos = pygame.Vector2(x, -20)
        
        if self._np_random.random() < 0.8: # 80% chance of fruit
            fruit_name = self._np_random.choice(list(self.FRUIT_TYPES.keys()))
            fruit_info = self.FRUIT_TYPES[fruit_name]
            self.fruits.append({
                "pos": pos,
                "type": fruit_name,
                "color": fruit_info["color"],
                "radius": fruit_info["radius"],
            })
        else: # 20% chance of bomb
            self.bombs.append({"pos": pos})

    def _check_slice_collision(self, circle_center, circle_radius):
        p1 = self.last_cursor_pos
        p2 = self.cursor_pos
        
        # Simplified check: if either end of the slice is inside the circle
        if p1.distance_to(circle_center) < circle_radius or p2.distance_to(circle_center) < circle_radius:
            return True

        # Full line-segment to circle collision
        d = p2 - p1
        f = p1 - circle_center
        
        if d.length_squared() == 0: # Slice is a point
            return f.length() < circle_radius

        t = max(0, min(1, (-f.dot(d)) / d.length_squared()))
        closest_point = p1 + t * d
        return closest_point.distance_to(circle_center) < circle_radius

    def _create_fruit_particles(self, fruit):
        for _ in range(20):
            angle = self._np_random.uniform(0, 2 * math.pi)
            speed = self._np_random.uniform(2, 6)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": fruit["pos"].copy(),
                "vel": vel,
                "color": fruit["color"],
                "size": self._np_random.integers(3, 7),
                "life": self._np_random.integers(20, 40)
            })
    
    def _create_bomb_particles(self, bomb):
        for _ in range(40):
            angle = self._np_random.uniform(0, 2 * math.pi)
            speed = self._np_random.uniform(3, 9)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            color = self._np_random.choice([(50,50,50), (255,100,0), (255,50,50)])
            self.particles.append({
                "pos": bomb["pos"].copy(),
                "vel": vel,
                "color": color,
                "size": self._np_random.integers(8, 16),
                "life": self._np_random.integers(30, 50)
            })

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 40))))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["pos"].x - p["size"]), int(p["pos"].y - p["size"])))

        # Render fruits
        for fruit in self.fruits:
            pos = (int(fruit["pos"].x), int(fruit["pos"].y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], fruit["radius"], fruit["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], fruit["radius"], fruit["color"])

        # Render bombs
        for bomb in self.bombs:
            pos = (int(bomb["pos"].x), int(bomb["pos"].y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BOMB_RADIUS, self.BOMB_COLOR)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BOMB_RADIUS, self.BOMB_COLOR)
            # Fuse
            fuse_end = bomb["pos"] + pygame.Vector2(0, -self.BOMB_RADIUS - 2)
            pygame.draw.line(self.screen, (100, 80, 50), bomb["pos"] + pygame.Vector2(0, -self.BOMB_RADIUS), fuse_end, 3)
            # Fuse spark
            if self.steps % 10 < 5:
                spark_pos = (int(fuse_end.x), int(fuse_end.y))
                pygame.gfxdraw.filled_circle(self.screen, spark_pos[0], spark_pos[1], 3, self.BOMB_FUSE_COLOR)
                
        # Render slice trails
        for trail in self.slice_trails:
            alpha = max(0, min(255, int(255 * (trail["life"] / 5.0))))
            color = (*self.COLOR_SLICE_TRAIL[:3], alpha)
            width = int(8 * (trail["life"] / 5.0))
            if width > 1:
                pygame.draw.line(self.screen, color, trail["start"], trail["end"], width)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Lives (as bomb icons)
        for i in range(self.lives):
            pos_x = self.WIDTH - 40 - (i * 40)
            pos_y = 30
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, 12, (150, 0, 0))
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, 12, (200, 0, 0))
            pygame.draw.line(self.screen, (100, 80, 50), (pos_x, pos_y - 12), (pos_x, pos_y - 16), 2)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    import os
    # Set the video driver to a dummy one if not rendering to screen
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- To run headlessly and check for errors ---
    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated:
    #         print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
    #         obs, info = env.reset()
    # env.close()
    # print("Headless test completed.")

    # --- To run with a window and play with keyboard ---
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Slicer")
    
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)

        # Convert the observation back for rendering
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS

        if terminated:
            print(f"Game Over! Score: {info['score']}")
            pygame.time.wait(2000) # Pause before closing
            
    env.close()