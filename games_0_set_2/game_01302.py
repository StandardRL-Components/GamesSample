
# Generated: 2025-08-27T16:41:44.214516
# Source Brief: brief_01302.md
# Brief Index: 1302

        
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
        "Controls: ←→ to move the catcher. Press space to activate the catcher and grab fruit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch the falling fruit before they hit the ground! Catch 25 to win, but miss 5 and you lose."
    )

    # Frames auto-advance for time-based gameplay
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    # Colors
    COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = (176, 224, 230) # Powder Blue
    COLOR_GROUND = (139, 69, 19)    # Saddle Brown
    COLOR_CATCHER = (0, 120, 255)   # Bright Blue
    COLOR_CATCHER_ACTIVE = (255, 255, 0) # Yellow
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)
    COLOR_MISS_X = (220, 20, 60) # Crimson
    
    FRUIT_COLORS = {
        "apple": (255, 0, 0),
        "lemon": (255, 255, 0),
        "orange": (255, 165, 0),
        "lime": (0, 255, 0),
        "plum": (148, 0, 211),
    }

    # Game parameters
    CATCHER_WIDTH = 80
    CATCHER_HEIGHT = 20
    CATCHER_SPEED = 10
    CATCHER_ACTIVE_DURATION = 8 # frames
    CATCHER_COOLDOWN = 12 # frames
    
    FRUIT_RADIUS = 12
    BASE_FRUIT_SPEED = 2.0
    FRUIT_SPAWN_RATE = 25 # frames
    
    WIN_CONDITION = 25
    LOSE_CONDITION = 5
    MAX_STEPS = 1800 # 60 seconds at 30 FPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Initialize state variables
        self.catcher_pos_x = 0
        self.catcher_active_timer = 0
        self.catcher_cooldown_timer = 0
        self.fruits = []
        self.particles = []
        self.splats = []
        self.steps = 0
        self.score = 0
        self.caught_count = 0
        self.missed_count = 0
        self.game_over = False
        self.fruit_spawn_timer = 0
        self.current_fruit_speed = 0.0
        
        self.reset()
        
        # self.validate_implementation() # Optional: call to check against spec

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.catcher_pos_x = self.SCREEN_WIDTH / 2
        self.catcher_active_timer = 0
        self.catcher_cooldown_timer = 0

        self.fruits = []
        self.particles = []
        self.splats = []
        
        self.steps = 0
        self.score = 0
        self.caught_count = 0
        self.missed_count = 0
        
        self.game_over = False
        self.fruit_spawn_timer = self.FRUIT_SPAWN_RATE
        self.current_fruit_speed = self.BASE_FRUIT_SPEED
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        self._handle_input(action)
        self._update_game_state()
        
        reward += self._check_collisions()
        reward += self._check_misses()
        
        self._update_difficulty()
        
        terminated = self._check_termination()
        
        if terminated:
            if self.caught_count >= self.WIN_CONDITION:
                reward += 50
            elif self.missed_count >= self.LOSE_CONDITION:
                reward -= 50
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, _ = action
        
        # Horizontal movement
        if movement == 3: # Left
            self.catcher_pos_x -= self.CATCHER_SPEED
        elif movement == 4: # Right
            self.catcher_pos_x += self.CATCHER_SPEED
        
        self.catcher_pos_x = np.clip(
            self.catcher_pos_x, self.CATCHER_WIDTH / 2, self.SCREEN_WIDTH - self.CATCHER_WIDTH / 2
        )
        
        # Activate catcher
        if space_pressed and self.catcher_cooldown_timer <= 0:
            self.catcher_active_timer = self.CATCHER_ACTIVE_DURATION
            self.catcher_cooldown_timer = self.CATCHER_COOLDOWN
            # sfx: whoosh sound

    def _update_game_state(self):
        # Decrement timers
        if self.catcher_active_timer > 0:
            self.catcher_active_timer -= 1
        if self.catcher_cooldown_timer > 0:
            self.catcher_cooldown_timer -= 1

        # Spawn new fruit
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            self.fruit_spawn_timer = self.FRUIT_SPAWN_RATE
            fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
            self.fruits.append({
                "x": self.np_random.uniform(self.FRUIT_RADIUS, self.SCREEN_WIDTH - self.FRUIT_RADIUS),
                "y": -self.FRUIT_RADIUS,
                "type": fruit_type,
                "color": self.FRUIT_COLORS[fruit_type],
            })

        # Move fruits
        for fruit in self.fruits:
            fruit["y"] += self.current_fruit_speed
            
        # Update particles
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.2 # Gravity
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        
        # Update splats
        for s in self.splats:
            s["lifespan"] -= 1
        self.splats = [s for s in self.splats if s["lifespan"] > 0]

    def _check_collisions(self):
        reward = 0
        if self.catcher_active_timer > 0:
            catch_rect = pygame.Rect(
                self.catcher_pos_x - self.CATCHER_WIDTH / 2,
                self.SCREEN_HEIGHT - self.CATCHER_HEIGHT - 10,
                self.CATCHER_WIDTH,
                self.CATCHER_HEIGHT + 10
            )
            
            fruits_to_remove = []
            for fruit in self.fruits:
                if catch_rect.collidepoint(fruit["x"], fruit["y"]):
                    fruits_to_remove.append(fruit)
                    self.caught_count += 1
                    self.score += 1
                    reward += 1
                    self._spawn_catch_particles(fruit["x"], fruit["y"], fruit["color"])
                    # sfx: catch success sound
            
            self.fruits = [f for f in self.fruits if f not in fruits_to_remove]
        return reward

    def _check_misses(self):
        reward = 0
        fruits_to_remove = []
        for fruit in self.fruits:
            if fruit["y"] > self.SCREEN_HEIGHT - self.CATCHER_HEIGHT / 2:
                fruits_to_remove.append(fruit)
                self.missed_count += 1
                reward -= 1
                self._spawn_splat(fruit["x"], fruit["color"])
                # sfx: splat sound
        
        self.fruits = [f for f in self.fruits if f not in fruits_to_remove]
        return reward

    def _update_difficulty(self):
        # Speed increases by 0.05 for every 5 fruits caught
        speed_increase_factor = self.caught_count // 5
        self.current_fruit_speed = self.BASE_FRUIT_SPEED + speed_increase_factor * 0.15

    def _check_termination(self):
        if self.caught_count >= self.WIN_CONDITION:
            return True
        if self.missed_count >= self.LOSE_CONDITION:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_ground()
        self._render_splats()
        self._render_fruits()
        self._render_catcher()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_ground(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.SCREEN_HEIGHT - 10, self.SCREEN_WIDTH, 10))

    def _render_catcher(self):
        catcher_rect = pygame.Rect(
            self.catcher_pos_x - self.CATCHER_WIDTH / 2,
            self.SCREEN_HEIGHT - self.CATCHER_HEIGHT,
            self.CATCHER_WIDTH,
            self.CATCHER_HEIGHT
        )
        # Draw catcher body
        pygame.draw.rect(self.screen, self.COLOR_CATCHER, catcher_rect, border_radius=5)
        
        # Draw active glow
        if self.catcher_active_timer > 0:
            glow_alpha = 100 + 155 * (self.catcher_active_timer / self.CATCHER_ACTIVE_DURATION)
            glow_surface = pygame.Surface((self.CATCHER_WIDTH + 20, self.CATCHER_HEIGHT + 20), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*self.COLOR_CATCHER_ACTIVE, glow_alpha), glow_surface.get_rect(), border_radius=15)
            self.screen.blit(glow_surface, (catcher_rect.x - 10, catcher_rect.y - 10))

    def _render_fruits(self):
        for fruit in self.fruits:
            pygame.gfxdraw.aacircle(self.screen, int(fruit["x"]), int(fruit["y"]), self.FRUIT_RADIUS, fruit["color"])
            pygame.gfxdraw.filled_circle(self.screen, int(fruit["x"]), int(fruit["y"]), self.FRUIT_RADIUS, fruit["color"])

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p["lifespan"] / 40))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, p["size"], p["size"]))
            self.screen.blit(temp_surf, (p["x"] - p["size"], p["y"] - p["size"]))

    def _render_splats(self):
        for s in self.splats:
            alpha = max(0, 200 * (s["lifespan"] / 60))
            pygame.gfxdraw.filled_ellipse(self.screen, s["x"], s["y"], s["rx"], s["ry"], (*s["color"], alpha))

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (15, 10))
        
        # Misses
        miss_text = "MISSES: "
        for i in range(self.LOSE_CONDITION):
            color = self.COLOR_MISS_X if i < self.missed_count else self.COLOR_TEXT_SHADOW
            x_char = self.font_ui.render("X", True, color)
            self.screen.blit(x_char, (self.SCREEN_WIDTH - 80 + i * 15, 10))
        
        # Game Over / Win message
        if self.game_over:
            if self.caught_count >= self.WIN_CONDITION:
                msg = "YOU WIN!"
            elif self.missed_count >= self.LOSE_CONDITION:
                msg = "GAME OVER"
            else:
                msg = "TIME'S UP!"
            
            large_font = pygame.font.SysFont("Arial", 60, bold=True)
            text_surf = large_font.render(msg, True, self.COLOR_TEXT)
            text_shadow = large_font.render(msg, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_shadow, (text_rect.x + 3, text_rect.y + 3))
            self.screen.blit(text_surf, text_rect)

    def _draw_text(self, text, pos):
        text_surf = self.font_ui.render(text, True, self.COLOR_TEXT)
        shadow_surf = self.font_ui.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
        self.screen.blit(text_surf, pos)

    def _spawn_catch_particles(self, x, y, color):
        for _ in range(15):
            self.particles.append({
                "x": x, "y": y,
                "vx": self.np_random.uniform(-2, 2),
                "vy": self.np_random.uniform(-4, -1),
                "lifespan": self.np_random.integers(20, 40),
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

    def _spawn_splat(self, x, color):
        self.splats.append({
            "x": int(x),
            "y": int(self.SCREEN_HEIGHT - 5),
            "rx": self.np_random.integers(15, 20),
            "ry": self.np_random.integers(5, 8),
            "color": color,
            "lifespan": 60
        })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "caught": self.caught_count,
            "missed": self.missed_count,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Catcher")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Convert observation back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}, Steps: {info['steps']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            print("--- Game Reset ---")
        
        clock.tick(env.FPS)
        
    env.close()