
# Generated: 2025-08-28T06:17:41.242154
# Source Brief: brief_02873.md
# Brief Index: 2873

        
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
        "Controls: Use arrow keys (↑↓←→) to move your blade and slice the falling fruit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit with a virtual blade to reach a target score before missing too many."
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
        self.font_large = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG_TOP = (10, 20, 30)
        self.COLOR_BG_BOTTOM = (40, 50, 60)
        self.COLOR_BLADE = (150, 255, 255)
        self.COLOR_BLADE_HOT = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEART = (255, 80, 80)
        self.FRUIT_COLORS = [
            (255, 60, 60),  # Apple Red
            (255, 200, 60), # Banana Yellow
            (60, 200, 60),  # Watermelon Green
            (255, 120, 0),  # Orange
        ]

        # Game parameters
        self.WIN_SCORE = 25
        self.MAX_MISSES = 3
        self.MAX_STEPS = 1000
        self.BLADE_SPEED = 15
        self.BLADE_TRAIL_LENGTH = 15
        self.PARTICLE_LIFESPAN = 30
        
        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.blade_pos = None
        self.last_blade_pos = None
        self.blade_trail = []
        self.fruits = []
        self.particles = []
        self.fruit_spawn_timer = 0
        self.base_fruit_speed = 2.0
        self.current_fruit_speed = 2.0
        
        # Initialize state
        self.reset()
        
        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        
        self.blade_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
        self.last_blade_pos = self.blade_pos.copy()
        self.blade_trail.clear()
        
        self.fruits.clear()
        self.particles.clear()
        
        self.fruit_spawn_timer = 30
        self.current_fruit_speed = self.base_fruit_speed
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01 # Small penalty for time passing
        
        self._handle_input(action)
        self._update_blade_trail()
        self._update_difficulty()
        self._spawn_fruits()

        sliced_count = self._update_fruits()
        self._update_particles()
        
        if sliced_count > 0:
            # Sound: play_slice_sound()
            self.score += sliced_count
            reward += sliced_count * 1.0
            
        self.steps += 1
        
        terminated = self._check_termination()
        
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            elif self.misses >= self.MAX_MISSES:
                reward -= 100 # Lose penalty
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        self.last_blade_pos = self.blade_pos.copy()
        
        if movement == 1: # Up
            self.blade_pos[1] -= self.BLADE_SPEED
        elif movement == 2: # Down
            self.blade_pos[1] += self.BLADE_SPEED
        elif movement == 3: # Left
            self.blade_pos[0] -= self.BLADE_SPEED
        elif movement == 4: # Right
            self.blade_pos[0] += self.BLADE_SPEED
            
        # Clamp blade position to screen bounds
        self.blade_pos[0] = np.clip(self.blade_pos[0], 0, self.WIDTH)
        self.blade_pos[1] = np.clip(self.blade_pos[1], 0, self.HEIGHT)

    def _update_blade_trail(self):
        self.blade_trail.append(self.blade_pos.copy())
        if len(self.blade_trail) > self.BLADE_TRAIL_LENGTH:
            self.blade_trail.pop(0)

    def _update_difficulty(self):
        self.current_fruit_speed = self.base_fruit_speed + 0.05 * (self.steps // 50)

    def _spawn_fruits(self):
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            x_pos = self.np_random.uniform(50, self.WIDTH - 50)
            x_vel = self.np_random.uniform(-1, 1)
            y_vel = self.current_fruit_speed + self.np_random.uniform(-0.5, 0.5)
            
            fruit = {
                "pos": np.array([x_pos, -30.0]),
                "vel": np.array([x_vel, y_vel]),
                "radius": self.np_random.integers(20, 30),
                "color": random.choice(self.FRUIT_COLORS),
                "angle": 0,
                "spin": self.np_random.uniform(-0.1, 0.1)
            }
            self.fruits.append(fruit)
            self.fruit_spawn_timer = self.np_random.integers(25, 40)

    def _update_fruits(self):
        sliced_count = 0
        for fruit in self.fruits[:]:
            fruit["pos"] += fruit["vel"]
            fruit["angle"] += fruit["spin"]

            # Check for slice
            if self._line_circle_intersect(self.last_blade_pos, self.blade_pos, fruit["pos"], fruit["radius"]):
                self._create_slice_particles(fruit)
                self.fruits.remove(fruit)
                sliced_count += 1
                continue
                
            # Check for miss
            if fruit["pos"][1] > self.HEIGHT + fruit["radius"]:
                # Sound: play_miss_sound()
                self.misses += 1
                self.fruits.remove(fruit)
        return sliced_count

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"][1] += 0.1 # Gravity
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True
        if self.misses >= self.MAX_MISSES:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / self.PARTICLE_LIFESPAN))
            color = (p["color"][0], p["color"][1], p["color"][2], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))

        # Render fruits
        for fruit in self.fruits:
            x, y = int(fruit["pos"][0]), int(fruit["pos"][1])
            r = fruit["radius"]
            
            # Glow effect
            glow_r = int(r * 1.5)
            glow_color = (fruit["color"][0], fruit["color"][1], fruit["color"][2], 50)
            temp_surf = pygame.Surface((glow_r*2, glow_r*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, glow_r, glow_r, glow_r, glow_color)
            self.screen.blit(temp_surf, (x - glow_r, y - glow_r))

            # Main fruit body
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, fruit["color"])
            pygame.gfxdraw.aacircle(self.screen, x, y, r, (0,0,0, 50))
            
            # Highlight
            highlight_r = r // 3
            highlight_x = x + int(r * 0.3 * math.cos(fruit["angle"] + math.pi/4))
            highlight_y = y - int(r * 0.3 * math.sin(fruit["angle"] + math.pi/4))
            pygame.gfxdraw.filled_circle(self.screen, highlight_x, highlight_y, highlight_r, (255,255,255,150))

        # Render blade trail
        for i, pos in enumerate(self.blade_trail):
            alpha = int(255 * (i / self.BLADE_TRAIL_LENGTH))
            size = int(8 * (i / self.BLADE_TRAIL_LENGTH))
            color = (
                self.COLOR_BLADE[0],
                self.COLOR_BLADE[1],
                self.COLOR_BLADE[2],
                alpha
            )
            if size > 1:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(pos[0] - size), int(pos[1] - size)))
        
        # Render blade "hot" tip
        if self.blade_trail:
            tip_pos = self.blade_trail[-1]
            pygame.gfxdraw.filled_circle(self.screen, int(tip_pos[0]), int(tip_pos[1]), 5, self.COLOR_BLADE_HOT)
            pygame.gfxdraw.aacircle(self.screen, int(tip_pos[0]), int(tip_pos[1]), 5, self.COLOR_BLADE_HOT)

    def _render_ui(self):
        # Score display
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Misses display (hearts)
        for i in range(self.MAX_MISSES - self.misses):
            self._draw_heart(self.WIDTH - 30 - (i * 35), 25, 20, self.COLOR_HEART)
            
        # Game Over text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                end_text = self.font_large.render("YOU WIN!", True, (100, 255, 100))
            else:
                end_text = self.font_large.render("GAME OVER", True, (255, 100, 100))
            
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
        }

    def _create_slice_particles(self, fruit):
        # Sound: play_splash_sound()
        num_particles = 20
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            particle = {
                "pos": fruit["pos"].copy(),
                "vel": vel,
                "lifespan": self.np_random.integers(self.PARTICLE_LIFESPAN // 2, self.PARTICLE_LIFESPAN),
                "radius": self.np_random.integers(2, 5),
                "color": fruit["color"]
            }
            self.particles.append(particle)

    def _line_circle_intersect(self, p1, p2, circle_pos, r):
        # If blade didn't move, just check if the point is in the circle
        if np.array_equal(p1, p2):
            return np.linalg.norm(p1 - circle_pos) < r

        d = p2 - p1
        f = p1 - circle_pos
        
        len_sq = np.dot(d, d)
        if len_sq == 0.0:
            return np.linalg.norm(f) < r

        t = max(0, min(1, -np.dot(f, d) / len_sq))
        closest_point = p1 + t * d
        distance = np.linalg.norm(closest_point - circle_pos)
        
        return distance < r
    
    def _draw_heart(self, x, y, size, color):
        s2 = size // 2
        s4 = size // 4
        # Two circles and a triangle to form a heart shape
        pygame.gfxdraw.filled_circle(self.screen, x - s4, y - s4, s4, color)
        pygame.gfxdraw.filled_circle(self.screen, x + s4, y - s4, s4, color)
        points = [(x - s2, y - s4), (x + s2, y - s4), (x, y + s2)]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

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
    # To run and visualize the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Slicer")
    
    running = True
    total_reward = 0
    
    # Mapping pygame keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4
    }
    
    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # This allows only one movement key at a time, which is fine
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0
            
        env.clock.tick(30) # Control the frame rate

    env.close()