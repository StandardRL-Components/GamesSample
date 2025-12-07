
# Generated: 2025-08-28T02:45:04.807701
# Source Brief: brief_01801.md
# Brief Index: 1801

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your slicer. Slice the fruit, avoid the bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game. Slice falling fruit to score points while avoiding bombs. "
        "The game gets faster over time. Reach 500 points to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.W, self.H = 640, 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24, bold=True)

        # Colors
        self.COLOR_BG_TOP = (15, 23, 42)
        self.COLOR_BG_BOTTOM = (51, 65, 85)
        self.COLOR_TRAIL = (255, 255, 255)
        self.COLOR_BOMB = (15, 23, 42)
        self.COLOR_BOMB_SKULL = (226, 232, 240)
        self.COLOR_BOMB_FUSE = (220, 38, 38)
        self.COLOR_TEXT = (241, 245, 249)

        self.FRUIT_TYPES = {
            "apple": {"color": (220, 38, 38), "radius": 15, "value": 1},
            "orange": {"color": (249, 115, 22), "radius": 16, "value": 2},
            "watermelon": {"color": (34, 197, 94), "radius": 20, "value": 3},
            "plum": {"color": (147, 51, 234), "radius": 12, "value": 1},
        }

        # Game parameters
        self.SLICER_SPEED = 15
        self.MAX_STEPS = 1500
        self.WIN_SCORE = 500
        self.MAX_BOMB_HITS = 3
        self.INITIAL_FALL_SPEED = 2.0
        self.FALL_SPEED_INCREASE = 0.02
        
        # Initialize state variables
        self.slicer_pos = None
        self.slicer_trail = None
        self.fruits = None
        self.bombs = None
        self.particles = None
        self.score = None
        self.bombs_hit = None
        self.steps = None
        self.game_over = None
        self.current_fall_speed = None
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.bombs_hit = 0
        self.game_over = False
        self.current_fall_speed = self.INITIAL_FALL_SPEED

        self.slicer_pos = np.array([self.W / 2, self.H / 2], dtype=np.float32)
        self.slicer_trail = deque(maxlen=10)
        self.fruits = []
        self.bombs = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False

        # --- 1. Handle Action ---
        movement = action[0]  # 0-4: none/up/down/left/right
        prev_slicer_pos = self.slicer_pos.copy()

        if movement == 1:  # Up
            self.slicer_pos[1] -= self.SLICER_SPEED
        elif movement == 2:  # Down
            self.slicer_pos[1] += self.SLICER_SPEED
        elif movement == 3:  # Left
            self.slicer_pos[0] -= self.SLICER_SPEED
        elif movement == 4:  # Right
            self.slicer_pos[0] += self.SLICER_SPEED
        
        if movement == 0:
            reward -= 0.01 # Small penalty for inactivity

        self.slicer_pos[0] = np.clip(self.slicer_pos[0], 0, self.W)
        self.slicer_pos[1] = np.clip(self.slicer_pos[1], 0, self.H)

        if np.linalg.norm(self.slicer_pos - prev_slicer_pos) > 1:
            self.slicer_trail.append(self.slicer_pos.copy())
        elif len(self.slicer_trail) > 0:
             # Fade out trail when not moving
            self.slicer_trail.popleft()

        # --- 2. Update Game State ---
        self._spawn_objects()
        self._update_particles()
        reward += self._update_fruits_and_bombs()
        
        # Only check for slices if the slicer moved
        if movement != 0:
             reward += self._handle_slicing(prev_slicer_pos)

        # --- 3. Difficulty Scaling ---
        if self.steps > 0 and self.steps % 100 == 0:
            self.current_fall_speed += self.FALL_SPEED_INCREASE

        # --- 4. Check Termination ---
        terminated = (self.bombs_hit >= self.MAX_BOMB_HITS) or \
                     (self.score >= self.WIN_SCORE) or \
                     (self.steps >= self.MAX_STEPS)

        if self.score >= self.WIN_SCORE and not self.game_over:
            reward += 100
        
        if terminated:
            self.game_over = True

        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_objects(self):
        # Spawn a new object on average every 20 frames
        if self.rng.random() < 0.05:
            x = self.rng.uniform(self.W * 0.1, self.W * 0.9)
            speed_multiplier = self.rng.uniform(0.9, 1.2)
            
            # 25% chance of being a bomb
            if self.rng.random() < 0.25:
                self.bombs.append({
                    "pos": np.array([x, -20.0]),
                    "vel": np.array([self.rng.uniform(-0.5, 0.5), self.current_fall_speed * speed_multiplier]),
                    "radius": 18
                })
            else:
                fruit_name = self.rng.choice(list(self.FRUIT_TYPES.keys()))
                fruit_info = self.FRUIT_TYPES[fruit_name]
                self.fruits.append({
                    "pos": np.array([x, -20.0]),
                    "vel": np.array([self.rng.uniform(-1, 1), self.current_fall_speed * speed_multiplier]),
                    "radius": fruit_info["radius"],
                    "type": fruit_name,
                    "value": fruit_info["value"],
                    "sliced": False,
                    "slice_angle": 0,
                    "slice_pos": None,
                    "angular_vel": self.rng.uniform(-0.1, 0.1),
                    "angle": 0
                })

    def _update_fruits_and_bombs(self):
        reward = 0
        
        # Update fruits
        for fruit in self.fruits:
            fruit["pos"] += fruit["vel"]
            fruit["angle"] += fruit["angular_vel"]
        
        # Update bombs
        for bomb in self.bombs:
            bomb["pos"] += bomb["vel"]

        # Cleanup off-screen objects
        initial_fruit_count = len(self.fruits)
        self.fruits = [f for f in self.fruits if f["pos"][1] < self.H + 50]
        missed_fruits = initial_fruit_count - len(self.fruits)
        reward -= missed_fruits * 1.0 # Penalty for missing fruit

        self.bombs = [b for b in self.bombs if b["pos"][1] < self.H + 50]
        
        return reward

    def _handle_slicing(self, prev_pos):
        reward = 0
        p1 = prev_pos
        p2 = self.slicer_pos
        
        # Avoid zero-length segment
        if np.array_equal(p1, p2):
            return 0
            
        # Slice fruits
        for fruit in self.fruits:
            if fruit["sliced"]:
                continue
            
            if self._line_circle_collision(p1, p2, fruit["pos"], fruit["radius"]):
                fruit["sliced"] = True
                fruit["slice_pos"] = fruit["pos"].copy()
                
                slice_vec = p2 - p1
                fruit["slice_angle"] = math.atan2(slice_vec[1], slice_vec[0])
                
                self.score += fruit["value"]
                reward += fruit["value"]
                
                self._create_juice_particles(fruit["pos"], self.FRUIT_TYPES[fruit["type"]]["color"])
                # sfx: "slice.wav"

        # Slice bombs
        for i in range(len(self.bombs) - 1, -1, -1):
            bomb = self.bombs[i]
            if self._line_circle_collision(p1, p2, bomb["pos"], bomb["radius"]):
                self.bombs_hit += 1
                reward -= 10
                self._create_explosion_particles(bomb["pos"])
                del self.bombs[i]
                # sfx: "explosion.wav"
        
        return reward

    def _line_circle_collision(self, p1, p2, circle_center, radius):
        # Using vector projection to find the closest point on the line segment to the circle center
        v = p2 - p1
        u = circle_center - p1
        t = np.dot(u, v) / np.dot(v, v)
        t_clamped = np.clip(t, 0, 1)
        closest_point = p1 + t_clamped * v
        distance = np.linalg.norm(closest_point - circle_center)
        return distance <= radius

    def _create_juice_particles(self, pos, color):
        for _ in range(20):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.rng.integers(20, 40),
                "color": color,
                "size": self.rng.uniform(2, 5)
            })

    def _create_explosion_particles(self, pos):
        for _ in range(40):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(2, 7)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            color = self.rng.choice([(249, 115, 22), (239, 68, 68), (100, 116, 139)])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.rng.integers(30, 60),
                "color": color,
                "size": self.rng.uniform(3, 8)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"][1] += 0.1  # Gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.H):
            interp = y / self.H
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.W, y))

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 60.0))))
            color_with_alpha = p["color"] + (alpha,)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])))

        # Render bombs
        for bomb in self.bombs:
            pos = (int(bomb["pos"][0]), int(bomb["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], bomb["radius"], self.COLOR_BOMB)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], bomb["radius"], self.COLOR_BOMB)
            # Skull
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], bomb["radius"]-5, self.COLOR_BOMB_SKULL)
            pygame.draw.circle(self.screen, self.COLOR_BOMB, (pos[0]-5, pos[1]-3), 2)
            pygame.draw.circle(self.screen, self.COLOR_BOMB, (pos[0]+5, pos[1]-3), 2)
            pygame.draw.line(self.screen, self.COLOR_BOMB, (pos[0]-3, pos[1]+4), (pos[0]+3, pos[1]+4), 2)


        # Render fruits
        for fruit in self.fruits:
            if not fruit["sliced"]:
                self._render_whole_fruit(fruit)
            else:
                self._render_sliced_fruit(fruit)
        
        # Render slicer trail
        if len(self.slicer_trail) > 1:
            points = [tuple(map(int, p)) for p in self.slicer_trail]
            pygame.draw.lines(self.screen, self.COLOR_TRAIL, False, points, 5)


    def _render_whole_fruit(self, fruit):
        pos = (int(fruit["pos"][0]), int(fruit["pos"][1]))
        radius = fruit["radius"]
        color = self.FRUIT_TYPES[fruit["type"]]["color"]
        
        # Simple shadow
        shadow_pos = (pos[0] + 3, pos[1] + 3)
        pygame.gfxdraw.filled_circle(self.screen, shadow_pos[0], shadow_pos[1], radius, (0,0,0,50))
        
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def _render_sliced_fruit(self, fruit):
        # When a fruit is sliced, we draw two halves moving apart
        # This is a simplified visual effect
        info = self.FRUIT_TYPES[fruit["type"]]
        color = info["color"]
        radius = info["radius"]
        
        offset_dist = 5
        
        # Vector perpendicular to the slice
        perp_angle = fruit["slice_angle"] + math.pi / 2
        offset_vec = np.array([math.cos(perp_angle), math.sin(perp_angle)]) * offset_dist
        
        pos1 = fruit["pos"] + offset_vec
        pos2 = fruit["pos"] - offset_vec
        
        # Draw two arcs to represent halves
        rect1 = pygame.Rect(pos1[0]-radius, pos1[1]-radius, radius*2, radius*2)
        rect2 = pygame.Rect(pos2[0]-radius, pos2[1]-radius, radius*2, radius*2)
        
        start_angle = fruit["slice_angle"] + math.pi
        end_angle = fruit["slice_angle"]

        pygame.draw.arc(self.screen, color, rect1, start_angle, end_angle, radius)
        pygame.draw.arc(self.screen, color, rect2, end_angle, start_angle, radius)

    def _render_ui(self):
        # Render score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Render bomb hits (lives)
        for i in range(self.MAX_BOMB_HITS):
            pos_x = self.W - 30 - (i * 35)
            pos_y = 30
            color = self.COLOR_BOMB if i >= self.bombs_hit else (150, 0, 0)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, 12, self.COLOR_BOMB)
            if i < self.bombs_hit:
                 # Draw a red X over used lives
                pygame.draw.line(self.screen, (220, 38, 38), (pos_x-8, pos_y-8), (pos_x+8, pos_y+8), 4)
                pygame.draw.line(self.screen, (220, 38, 38), (pos_x-8, pos_y+8), (pos_x+8, pos_y-8), 4)
        
        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                end_text = "YOU WIN!"
            else:
                end_text = "GAME OVER"
            
            text_surface = self.font_large.render(end_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(text_surface, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bombs_hit": self.bombs_hit,
        }
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Fruit Slicer")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        movement = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            # The action space requires all 3 components
            action = [movement, 0, 0] # Space and Shift are not used
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            # Wait a bit after game over before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        clock.tick(30) # Run at 30 FPS
        
    env.close()