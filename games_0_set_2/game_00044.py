import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode, which is required for Gymnasium environments
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A fast-paced pixel art platformer where players race against time to reach the flag.
    The core mechanic involves balancing risky jumps for bonus points against safer, slower progress.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to run, and ↑ to jump. Reach the green flag before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced pixel platformer. Jump across gaps to reach the goal. Bigger jumps give more points!"
    )

    # Frames auto-advance for real-time physics and timer.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game Constants ---
        self.FPS = 30
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -14
        self.PLAYER_SPEED = 6
        self.FRICTION = 0.85
        self.MAX_TIME = 180 * self.FPS  # 180 seconds

        # --- Colors ---
        self.COLOR_BG_TOP = (127, 219, 255)
        self.COLOR_BG_BOTTOM = (0, 116, 217)
        self.COLOR_PLAYER = (255, 65, 54)
        self.COLOR_PLATFORM = (170, 170, 170)
        self.COLOR_PLATFORM_SHADOW = (85, 85, 85)
        self.COLOR_FLAG_POLE = (120, 80, 40)
        self.COLOR_FLAG = (46, 204, 64)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)
        self.COLOR_PARTICLE = (220, 220, 220)

        # --- State Variables (initialized in reset) ---
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.player_squash = None
        self.is_on_ground = None
        self.last_jump_start_x = None
        self.platforms = None
        self.flag_rect = None
        self.flag_pole_rect = None
        self.particles = None
        self.steps = None
        self.score = None
        self.timer = None
        self.game_over = None
        self.game_over_message = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.timer = self.MAX_TIME
        self.game_over = False
        self.game_over_message = ""

        self.player_pos = pygame.Vector2(80, 300)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 24, 24)
        self.player_rect.center = self.player_pos
        self.player_squash = pygame.Vector2(1, 1)

        self.is_on_ground = False
        self.last_jump_start_x = self.player_pos.x

        self.particles = []
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        # Starting platform
        start_platform = pygame.Rect(10, 350, 150, 50)
        self.platforms.append(start_platform)

        current_x = start_platform.right
        
        # Procedurally generate middle platforms
        while current_x < self.screen_width - 200:
            stage = 1 + (self.MAX_TIME - self.timer) // (60 * self.FPS)
            max_gap = 40 + (stage - 1) * 10
            min_gap = 10
            
            gap = self.np_random.integers(min_gap, max_gap + 1)
            width = self.np_random.integers(40, 100 + 1)
            
            current_x += gap
            new_platform = pygame.Rect(current_x, 350, width, 50)
            self.platforms.append(new_platform)
            current_x += width

        # Final platform with flag
        last_platform = self.platforms[-1]
        self.flag_pole_rect = pygame.Rect(last_platform.centerx - 5, last_platform.top - 60, 10, 60)
        self.flag_rect = pygame.Rect(self.flag_pole_rect.left - 30, self.flag_pole_rect.top, 30, 20)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # --- Handle Input ---
        movement = action[0]
        
        if movement == 3:  # Left
            self.player_vel.x -= 1.5
        elif movement == 4:  # Right
            self.player_vel.x += 1.5
            reward += 0.01 # Small reward for moving towards goal
        
        if movement == 1 and self.is_on_ground:  # Jump
            self.player_vel.y = self.JUMP_STRENGTH
            self.is_on_ground = False
            self.last_jump_start_x = self.player_pos.x
            self.player_squash.x = 0.8
            self.player_squash.y = 1.2
            # sfx: jump_sound()

        # --- Update Game Logic ---
        self.steps += 1
        self.timer -= 1
        
        if self.player_vel.length() < 0.1:
            reward -= 0.01 # Small penalty for being idle
        
        # Physics
        self.player_vel.y += self.GRAVITY
        self.player_vel.x *= self.FRICTION
        if abs(self.player_vel.x) > self.PLAYER_SPEED:
            self.player_vel.x = math.copysign(self.PLAYER_SPEED, self.player_vel.x)

        self.player_pos += self.player_vel
        
        # Animation
        self.player_squash.x = max(0.5, min(1.5, self.player_squash.x + (1 - self.player_squash.x) * 0.2))
        self.player_squash.y = max(0.5, min(1.5, self.player_squash.y + (1 - self.player_squash.y) * 0.2))

        # Keep player on screen horizontally
        if self.player_pos.x < self.player_rect.width / 2:
            self.player_pos.x = self.player_rect.width / 2
            self.player_vel.x = 0
        if self.player_pos.x > self.screen_width - self.player_rect.width / 2:
            self.player_pos.x = self.screen_width - self.player_rect.width / 2
            self.player_vel.x = 0

        self.player_rect.center = self.player_pos

        # Collision Detection
        was_on_ground = self.is_on_ground
        self.is_on_ground = False
        for plat in self.platforms:
            if self.player_rect.colliderect(plat) and self.player_vel.y > 0 and self.player_rect.bottom - self.player_vel.y < plat.top + 1:
                self.player_rect.bottom = plat.top
                self.player_pos.y = self.player_rect.centery
                self.player_vel.y = 0
                self.is_on_ground = True
                if not was_on_ground: # Just landed
                    self.player_squash.x = 1.3
                    self.player_squash.y = 0.7
                    self._spawn_particles(self.player_rect.midbottom, 10)
                    # sfx: land_sound()
                    
                    jump_dist = abs(self.player_pos.x - self.last_jump_start_x)
                    if jump_dist > 20:
                        reward += 1.0 # Risky jump bonus
                    elif jump_dist < 10 and self.last_jump_start_x is not None:
                        reward -= 0.2 # Safe jump penalty

        # --- Termination Conditions ---
        terminated = False
        # 1. Fell in a pit
        if self.player_pos.y > self.screen_height:
            reward -= 100
            terminated = True
            self.game_over_message = "YOU FELL!"
            # sfx: fail_sound()

        # 2. Reached the flag
        if self.player_rect.colliderect(self.flag_rect):
            reward += 100
            terminated = True
            self.game_over_message = "YOU WIN!"
            # sfx: win_sound()

        # 3. Ran out of time
        if self.timer <= 0:
            reward -= 50
            terminated = True
            self.game_over_message = "TIME'S UP!"
            # sfx: timeout_sound()

        if terminated:
            self.game_over = True
            
        self.score += reward

        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # --- Draw Background ---
        for y in range(self.screen_height):
            interp = y / self.screen_height
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))

        # --- Draw Game Elements ---
        # Platforms
        for plat in self.platforms:
            shadow_rect = plat.copy()
            shadow_rect.height = 8
            shadow_rect.top = plat.bottom - 8
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_SHADOW, shadow_rect)
            main_rect = plat.copy()
            main_rect.height = plat.height - 8
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, main_rect)
        
        # Flag
        pygame.draw.rect(self.screen, self.COLOR_FLAG_POLE, self.flag_pole_rect)
        pygame.draw.rect(self.screen, self.COLOR_FLAG, self.flag_rect)
        
        # Particles
        for p in self.particles:
            size = max(0, int(p['size'] * (p['life'] / p['max_life'])))
            if size > 0:
                pygame.draw.rect(self.screen, self.COLOR_PARTICLE, (*p['pos'], size, size))

        # Player
        squashed_width = int(self.player_rect.width * self.player_squash.x)
        squashed_height = int(self.player_rect.height * self.player_squash.y)
        display_rect = pygame.Rect(0, 0, squashed_width, squashed_height)
        display_rect.center = self.player_rect.center
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, display_rect, border_radius=4)
        
        # --- Draw UI ---
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (15, 10), self.font_ui)
        
        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_text = f"TIME: {time_left:.2f}"
        self._draw_text(timer_text, (self.screen_width - 165, 10), self.font_ui)
        
        # Game Over Message
        if self.game_over:
            self._draw_text(self.game_over_message, self.screen.get_rect().center, self.font_game_over, center=True)

    def _draw_text(self, text, pos, font, color=None, shadow_color=None, center=False):
        if color is None:
            color = self.COLOR_TEXT
        if shadow_color is None:
            shadow_color = self.COLOR_TEXT_SHADOW

        text_surface = font.render(text, True, color)
        shadow_surface = font.render(text, True, shadow_color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        shadow_pos = (text_rect.left + 2, text_rect.top + 2)
        self.screen.blit(shadow_surface, shadow_pos)
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.timer / self.FPS)
        }
        
    def _spawn_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'size': self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and visualize the environment
if __name__ == "__main__":
    # The environment is created in headless mode, which is standard for training.
    # To visualize, we need to re-initialize pygame with a display.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS", etc.
    
    env = GameEnv()
    
    # --- For manual play ---
    # To control, you need to map keyboard keys to the MultiDiscrete action space.
    # This is a simple mapping for demonstration.
    # A real human_playing.py would be more robust.
    
    obs, info = env.reset()
    done = False
    
    # Re-initialize pygame for display
    pygame.display.init()
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Pixel Platformer")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    running = True
    while running:
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        # No down action
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if not env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                print(f"Episode finished! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()