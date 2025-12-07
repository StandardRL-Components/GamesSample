
# Generated: 2025-08-27T16:05:27.791292
# Source Brief: brief_01114.md
# Brief Index: 1114

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the slicer. Press spacebar to slice horizontally, and shift to slice vertically. Slice fruits for points and avoid the bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you slice falling fruit to score points while dodging explosive bombs. Precision and timing are key to achieving a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        self.font_small = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 64)

        # Colors
        self.COLOR_BG_TOP = (4, 0, 38)
        self.COLOR_BG_BOTTOM = (44, 0, 54)
        self.COLOR_SLICER_CURSOR = (220, 220, 220)
        self.COLOR_SLICE_LINE = (255, 255, 255)
        self.COLOR_BOMB = (10, 10, 10)
        self.COLOR_TEXT = (255, 255, 220)
        self.FRUIT_COLORS = {
            "apple": (255, 50, 50),
            "orange": (255, 165, 0),
            "banana": (255, 255, 50),
        }

        # Game parameters
        self.SLICER_SPEED = 12
        self.SLICE_LENGTH = 120
        self.SLICE_DURATION = 5  # frames
        self.MAX_STEPS = 10000
        self.WIN_SCORE = 1000
        self.INITIAL_LIVES = 3
        self.INITIAL_SPAWN_RATE = 1.0 # objects per second
        self.SPAWN_RATE_INCREASE = 0.1
        self.SPAWN_RATE_INTERVAL = 200 # steps

        # Initialize state variables
        self.slicer_pos = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.fruits = None
        self.bombs = None
        self.particles = None
        self.slice_info = None
        self.steps = None
        self.score = None
        self.lives = None
        self.game_over = None
        self.spawn_timer = None
        self.spawn_rate = None

        self.reset()
        # self.validate_implementation() # Commented out for final submission as per instructions.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.slicer_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.prev_space_held = False
        self.prev_shift_held = False
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.slice_info = {"active": False, "timer": 0, "p1": (0, 0), "p2": (0, 0)}
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.spawn_timer = 0.0
        self.spawn_rate = self.INITIAL_SPAWN_RATE

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # 1. Handle player input and slicer movement
        self._handle_input(movement, space_held, shift_held)

        # 2. Update game state
        reward += self._update_slice()
        self._update_objects()
        self._update_particles()
        self._spawn_objects()

        # 3. Update difficulty
        self.steps += 1
        if self.steps > 0 and self.steps % self.SPAWN_RATE_INTERVAL == 0:
            self.spawn_rate += self.SPAWN_RATE_INCREASE

        # 4. Check for termination
        terminated = False
        if self.lives <= 0:
            terminated = True
            reward -= 100
            # Sound: game_over.wav
        elif self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100
            # Sound: win.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Move slicer
        if movement == 1: self.slicer_pos[1] -= self.SLICER_SPEED  # Up
        if movement == 2: self.slicer_pos[1] += self.SLICER_SPEED  # Down
        if movement == 3: self.slicer_pos[0] -= self.SLICER_SPEED  # Left
        if movement == 4: self.slicer_pos[0] += self.SLICER_SPEED  # Right

        self.slicer_pos[0] = np.clip(self.slicer_pos[0], 0, self.WIDTH)
        self.slicer_pos[1] = np.clip(self.slicer_pos[1], 0, self.HEIGHT)

        # Activate slice on key press (rising edge)
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        if not self.slice_info["active"]:
            if space_pressed:
                # Horizontal Slice
                self.slice_info["active"] = True
                self.slice_info["timer"] = self.SLICE_DURATION
                p1 = (self.slicer_pos[0] - self.SLICE_LENGTH / 2, self.slicer_pos[1])
                p2 = (self.slicer_pos[0] + self.SLICE_LENGTH / 2, self.slicer_pos[1])
                self.slice_info["p1"] = np.array(p1)
                self.slice_info["p2"] = np.array(p2)
                # Sound: slice_whoosh.wav
            elif shift_pressed:
                # Vertical Slice
                self.slice_info["active"] = True
                self.slice_info["timer"] = self.SLICE_DURATION
                p1 = (self.slicer_pos[0], self.slicer_pos[1] - self.SLICE_LENGTH / 2)
                p2 = (self.slicer_pos[0], self.slicer_pos[1] + self.SLICE_LENGTH / 2)
                self.slice_info["p1"] = np.array(p1)
                self.slice_info["p2"] = np.array(p2)
                # Sound: slice_whoosh.wav
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_slice(self):
        reward = 0
        if not self.slice_info["active"]:
            return reward

        self.slice_info["timer"] -= 1
        if self.slice_info["timer"] <= 0:
            self.slice_info["active"] = False

        # Check for collisions with fruits
        sliced_fruits = []
        for fruit in self.fruits:
            if self._check_line_circle_collision(self.slice_info["p1"], self.slice_info["p2"], fruit["pos"], fruit["radius"]):
                sliced_fruits.append(fruit)
                reward += 1
                self.score += fruit["points"]
                self._create_particles(fruit["pos"], self.FRUIT_COLORS[fruit["type"]], 30)
                # Sound: fruit_slice.wav
        self.fruits = [f for f in self.fruits if f not in sliced_fruits]

        # Check for collisions with bombs
        hit_bombs = []
        for bomb in self.bombs:
            if self._check_line_circle_collision(self.slice_info["p1"], self.slice_info["p2"], bomb["pos"], bomb["radius"]):
                hit_bombs.append(bomb)
                reward -= 5
                self.lives -= 1
                self._create_particles(bomb["pos"], (255, 60, 0), 50, is_explosion=True)
                # Sound: explosion.wav
        self.bombs = [b for b in self.bombs if b not in hit_bombs]
        
        return reward

    def _update_objects(self):
        # Update fruits
        for fruit in self.fruits:
            fruit["pos"] += fruit["vel"]
        self.fruits = [f for f in self.fruits if f["pos"][1] < self.HEIGHT + f["radius"]]

        # Update bombs
        for bomb in self.bombs:
            bomb["pos"] += bomb["vel"]
        self.bombs = [b for b in self.bombs if b["pos"][1] < self.HEIGHT + b["radius"]]

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _spawn_objects(self):
        # FPS is assumed to be 30 for auto_advance=True
        self.spawn_timer += 1 / 30.0
        if self.spawn_timer > 1.0 / self.spawn_rate:
            self.spawn_timer = 0
            
            x_pos = random.uniform(50, self.WIDTH - 50)
            vel_x = random.uniform(-1, 1)
            vel_y = random.uniform(2, 5)
            
            if random.random() < 0.2: # 20% chance of bomb
                self.bombs.append({
                    "pos": np.array([x_pos, -20.0]),
                    "vel": np.array([vel_x, vel_y]),
                    "radius": 15,
                })
            else: # 80% chance of fruit
                fruit_type = random.choice(list(self.FRUIT_COLORS.keys()))
                radius = random.randint(15, 25)
                self.fruits.append({
                    "pos": np.array([x_pos, -float(radius)]),
                    "vel": np.array([vel_x, vel_y * (20 / radius)]), # Smaller fruits fall faster
                    "radius": radius,
                    "type": fruit_type,
                    "points": int(500 / radius) # Smaller fruits worth more
                })

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            color_ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[0] * color_ratio,
                self.COLOR_BG_TOP[1] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[1] * color_ratio,
                self.COLOR_BG_TOP[2] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[2] * color_ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / p["initial_life"]))))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), color)

        # Render fruits
        for fruit in self.fruits:
            pos = (int(fruit["pos"][0]), int(fruit["pos"][1]))
            radius = fruit["radius"]
            color = self.FRUIT_COLORS[fruit["type"]]
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        # Render bombs
        for bomb in self.bombs:
            pos = (int(bomb["pos"][0]), int(bomb["pos"][1]))
            radius = bomb["radius"]
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_BOMB)
            # Fuse
            fuse_end = (pos[0] + 5, pos[1] - radius - 2)
            pygame.draw.line(self.screen, (150, 150, 150), (pos[0], pos[1] - radius), fuse_end, 2)
            # Spark
            if self.steps % 4 < 2:
                 pygame.draw.circle(self.screen, (255, 255, 0), fuse_end, 2)

        # Render slicer cursor
        cursor_pos = (int(self.slicer_pos[0]), int(self.slicer_pos[1]))
        pygame.draw.line(self.screen, self.COLOR_SLICER_CURSOR, (cursor_pos[0] - 5, cursor_pos[1]), (cursor_pos[0] + 5, cursor_pos[1]), 1)
        pygame.draw.line(self.screen, self.COLOR_SLICER_CURSOR, (cursor_pos[0], cursor_pos[1] - 5), (cursor_pos[0], cursor_pos[1] + 5), 1)

        # Render active slice
        if self.slice_info["active"]:
            p1 = (int(self.slice_info["p1"][0]), int(self.slice_info["p1"][1]))
            p2 = (int(self.slice_info["p2"][0]), int(self.slice_info["p2"][1]))
            # Glow effect
            alpha = 255 * (self.slice_info["timer"] / self.SLICE_DURATION)
            for i in range(4, 0, -1):
                pygame.draw.line(self.screen, (*self.COLOR_SLICE_LINE, alpha/4), p1, p2, i * 2 + 2)
            pygame.draw.line(self.screen, self.COLOR_SLICE_LINE, p1, p2, 2)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_small.render(f"Lives: {self.lives}", True, self.COLOR_TEXT)
        text_rect = lives_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(lives_text, text_rect)
        
        # Game Over / Win message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def _create_particles(self, pos, color, count, is_explosion=False):
        for _ in range(count):
            if is_explosion:
                speed = random.uniform(3, 8)
                life = random.randint(20, 40)
                radius = random.uniform(2, 5)
                p_color = random.choice([(255, 60, 0), (255, 165, 0), (255, 255, 0)])
            else:
                speed = random.uniform(1, 4)
                life = random.randint(15, 30)
                radius = random.uniform(1, 3)
                p_color = color

            angle = random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": life,
                "initial_life": life,
                "color": p_color,
                "radius": radius
            })

    def _check_line_circle_collision(self, p1, p2, circle_center, circle_radius):
        # Vector from p1 to p2
        line_vec = p2 - p1
        # Vector from p1 to circle center
        p1_to_circle = circle_center - p1
        
        # Project p1_to_circle onto the line to find the closest point on the infinite line
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0: # p1 and p2 are the same point
            return np.linalg.norm(p1_to_circle) < circle_radius

        t = np.dot(p1_to_circle, line_vec) / line_len_sq
        t = np.clip(t, 0, 1) # Clamp to the line segment

        closest_point = p1 + t * line_vec
        distance = np.linalg.norm(circle_center - closest_point)
        
        return distance < circle_radius

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
        print("Running implementation validation...")
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
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need a Pygame screen
    pygame.display.set_caption("Fruit Slicer")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        # Map keys to action space
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset() # Automatically reset for continuous play

        # Render the observation to the display screen
        # The observation is (H, W, C), but pygame surfaces are (W, H)
        # Transpose is needed to swap W and H for pygame's surfarray
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control the frame rate
        env.clock.tick(30)
        
    env.close()