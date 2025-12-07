
# Generated: 2025-08-28T03:44:50.184067
# Source Brief: brief_02111.md
# Brief Index: 2111

        
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


# Helper classes for game objects
class Fruit:
    def __init__(self, pos, vel, radius, color, fruit_type):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.type = fruit_type
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-2, 2)

    def update(self):
        self.pos += self.vel
        self.rotation = (self.rotation + self.rotation_speed) % 360

class Particle:
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.vel *= 0.98  # friction
        self.lifespan -= 1
        self.radius = max(0, self.radius * 0.98)


class SliceTrail:
    def __init__(self, start_pos, end_pos, lifespan=10):
        self.start_pos = pygame.Vector2(start_pos)
        self.end_pos = pygame.Vector2(end_pos)
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.lifespan -= 1


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Use arrow keys to move the cursor. Press space to slice."
    game_description = "A fast-paced arcade game. Slice the falling fruit to score points, but be careful! Missing 5 fruits ends the game."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    CURSOR_SPEED = 15
    SLICE_RADIUS = 25
    MAX_STEPS = 1800  # 60 seconds at 30fps
    TARGET_SCORE = 500
    MAX_LIVES = 5
    FPS = 30

    # --- Colors ---
    COLOR_BG_START = (20, 30, 40)
    COLOR_BG_END = (60, 40, 30)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SLICE = (220, 220, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    FRUIT_TYPES = {
        "apple": {"color": (220, 20, 60), "radius": 18},
        "orange": {"color": (255, 140, 0), "radius": 20},
        "lemon": {"color": (255, 235, 59), "radius": 16},
        "lime": {"color": (139, 195, 74), "radius": 16},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.font_combo = pygame.font.Font(None, 32)
        
        self.np_random = None
        self.cursor_pos = None
        self.fruits = None
        self.particles = None
        self.slice_trails = None
        self.score = None
        self.lives = None
        self.steps = None
        self.game_over = None
        self.combo_count = None
        self.last_space_state = None
        self.spawn_timer = None
        self.initial_spawn_period = 90  # 3 seconds
        self.current_spawn_period = None
        self.cursor_trail = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             self.np_random = np.random.default_rng(seed=seed)
        else:
             self.np_random = np.random.default_rng()

        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.cursor_trail = [self.cursor_pos.copy() for _ in range(10)]
        self.fruits = []
        self.particles = []
        self.slice_trails = []
        self.score = 0
        self.lives = self.MAX_LIVES
        self.steps = 0
        self.game_over = False
        self.combo_count = 0
        self.last_space_state = False
        self.current_spawn_period = self.initial_spawn_period
        self.spawn_timer = self.current_spawn_period

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False
        fruit_sliced_this_step = False

        # --- 1. Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        is_slicing = space_held and not self.last_space_state
        self.last_space_state = space_held
        
        prev_cursor_pos = self.cursor_pos.copy()
        self._update_cursor(movement)

        # --- 2. Update Game State ---
        self._update_lists() # Updates particles, slice trails

        # --- 3. Handle Slicing ---
        if is_slicing:
            # sfx: slice_whoosh.wav
            self.slice_trails.append(SliceTrail(prev_cursor_pos, self.cursor_pos))
            sliced_fruits_this_action = []
            
            for fruit in self.fruits[:]:
                if self.cursor_pos.distance_to(fruit.pos) < fruit.radius + self.SLICE_RADIUS:
                    sliced_fruits_this_action.append(fruit)

            if sliced_fruits_this_action:
                fruit_sliced_this_step = True
                for fruit in sliced_fruits_this_action:
                    # sfx: fruit_squish.wav
                    reward += 1.0  # Base reward for slicing
                    self.score += 10
                    self.combo_count += 1
                    if self.combo_count > 1:
                        reward += 5.0 # Combo reward
                        self.score += 5 * (self.combo_count - 1) # Score bonus for combos
                    self._create_explosion(fruit.pos, fruit.color)
                    self.fruits.remove(fruit)
            else: # Slice missed
                self.combo_count = 0

        # --- 4. Update Fruits & Handle Misses ---
        missed_fruit_count = self._update_fruits()
        if missed_fruit_count > 0:
            # sfx: miss.wav
            self.lives -= missed_fruit_count
            self.combo_count = 0
            # No direct reward penalty for miss, handled by terminal condition

        # --- 5. Apply Other Rewards/Penalties ---
        reward -= 0.1 * len(self.fruits)  # Penalty for letting fruit linger
        
        if movement != 0 and not fruit_sliced_this_step:
            reward -= 0.1 # Small penalty for movement without purpose
        
        # --- 6. Spawn New Fruits ---
        self._spawn_fruit()

        # --- 7. Check for Termination ---
        self.steps += 1
        if self.lives <= 0:
            terminated = True
            reward -= 100
            self.game_over = True
        elif self.score >= self.TARGET_SCORE:
            terminated = True
            reward += 100
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_cursor(self, movement):
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        self.cursor_trail.pop(0)
        self.cursor_trail.append(self.cursor_pos.copy())

    def _update_lists(self):
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0: self.particles.remove(p)
        for s in self.slice_trails[:]:
            s.update()
            if s.lifespan <= 0: self.slice_trails.remove(s)

    def _update_fruits(self):
        missed_count = 0
        for fruit in self.fruits[:]:
            fruit.update()
            if fruit.pos.y - fruit.radius > self.HEIGHT:
                self.fruits.remove(fruit)
                missed_count += 1
        return missed_count

    def _spawn_fruit(self):
        self.spawn_timer -= 1
        # Difficulty scaling for spawn rate
        spawn_rate_progress = self.steps / self.MAX_STEPS
        self.current_spawn_period = self.initial_spawn_period - (self.initial_spawn_period - 15) * spawn_rate_progress

        if self.spawn_timer <= 0:
            self.spawn_timer = self.current_spawn_period
            
            fruit_key = self.np_random.choice(list(self.FRUIT_TYPES.keys()))
            f_type = self.FRUIT_TYPES[fruit_key]
            
            x_pos = self.np_random.uniform(f_type["radius"], self.WIDTH - f_type["radius"])
            y_pos = -f_type["radius"]
            
            # Difficulty scaling for speed
            speed_progress = self.steps / self.MAX_STEPS
            base_y_speed = 2.0 + 3.0 * speed_progress
            
            vx = self.np_random.uniform(-1.5, 1.5)
            vy = base_y_speed + self.np_random.uniform(-0.5, 0.5)

            self.fruits.append(Fruit((x_pos, y_pos), (vx, vy), f_type["radius"], f_type["color"], fruit_key))

    def _create_explosion(self, pos, color):
        num_particles = 20
        for _ in range(num_particles):
            vel = pygame.Vector2(self.np_random.uniform(-4, 4), self.np_random.uniform(-4, 4))
            radius = self.np_random.uniform(2, 6)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append(Particle(pos, vel, radius, color, lifespan))

    def _get_observation(self):
        # --- Background ---
        progress = min(1, self.score / self.TARGET_SCORE)
        bg_color = self._lerp_color(self.COLOR_BG_START, self.COLOR_BG_END, progress)
        self.screen.fill(bg_color)

        # --- Render Game Elements ---
        self._render_game()

        # --- Render UI ---
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render slice trails
        for trail in self.slice_trails:
            alpha = int(255 * (trail.lifespan / trail.max_lifespan))
            color = (*self.COLOR_SLICE, alpha)
            pygame.draw.line(self.screen, color, trail.start_pos, trail.end_pos, 8)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p.lifespan / p.max_lifespan))
            color = (*p.color, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), int(p.radius), color)

        # Render fruits
        for fruit in self.fruits:
            pygame.gfxdraw.filled_circle(self.screen, int(fruit.pos.x), int(fruit.pos.y), fruit.radius, fruit.color)
            pygame.gfxdraw.aacircle(self.screen, int(fruit.pos.x), int(fruit.pos.y), fruit.radius, (0,0,0,50))


        # Render cursor trail
        for i, pos in enumerate(self.cursor_trail):
            alpha = int(255 * (i / len(self.cursor_trail)))
            radius = int(3 + 10 * (i / len(self.cursor_trail)))
            color = (*self.COLOR_CURSOR, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, color)
            
        # Render cursor
        pygame.gfxdraw.filled_circle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 12, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 12, (0,0,0, 100))


    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, shadow_color):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score
        score_text = f"Score: {self.score}"
        draw_text(score_text, self.font_large, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        
        # Lives
        lives_text = f"Lives: {self.lives}"
        text_w = self.font_large.size(lives_text)[0]
        draw_text(lives_text, self.font_large, self.COLOR_TEXT, (self.WIDTH - text_w - 10, 10), self.COLOR_TEXT_SHADOW)

        # Combo
        if self.combo_count > 1:
            combo_text = f"{self.combo_count}x COMBO!"
            text_w, text_h = self.font_combo.size(combo_text)
            draw_text(combo_text, self.font_combo, (255, 255, 100), (self.WIDTH/2 - text_w/2, 60), self.COLOR_TEXT_SHADOW)
        
        if self.game_over:
            if self.score >= self.TARGET_SCORE:
                end_text = "YOU WIN!"
            else:
                end_text = "GAME OVER"
            text_w, text_h = self.font_large.size(end_text)
            draw_text(end_text, self.font_large, (255, 200, 200), (self.WIDTH/2 - text_w/2, self.HEIGHT/2 - text_h/2), self.COLOR_TEXT_SHADOW)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "combo": self.combo_count,
        }
        
    def _lerp_color(self, c1, c2, t):
        t = max(0, min(1, t))
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Fruit Slicer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already the rendered screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()