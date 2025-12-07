
# Generated: 2025-08-28T03:53:58.441115
# Source Brief: brief_05082.md
# Brief Index: 5082

        
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
        "Controls: ←→ to move, ↑ or Space to jump. Reach the top before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade platformer. Jump between rising platforms to reach the summit before the timer expires."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (30, 0, 50)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255, 50)
    NEON_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
        (0, 255, 128),  # Spring Green
    ]
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_SHADOW = (20, 20, 20)

    # Physics
    PLAYER_SIZE = 16
    PLAYER_SPEED = 5.0
    GRAVITY = 0.8
    JUMP_STRENGTH = -12.0
    
    # Game Mechanics
    GOAL_Y = 50  # Reach this y-coordinate to win
    PLATFORM_HEIGHT = 10
    INITIAL_PLATFORM_SPEED = 2.0 / FPS # 2 pixels/sec
    PLATFORM_SPEED_INCREASE = (0.05 / FPS) / 500 # 0.05 px/sec increase per 500 steps

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        
        # Internal state variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.on_ground = None
        self.platforms = None
        self.particles = None
        self.steps = None
        self.score = None
        self.timer = None
        self.platform_speed = None
        self.highest_platform_idx = None
        
        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize player state
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_rect.center = self.player_pos
        self.on_ground = False

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.timer = self.MAX_STEPS / self.FPS
        self.platform_speed = self.INITIAL_PLATFORM_SPEED
        self.highest_platform_idx = 0
        
        # Initialize dynamic elements
        self.platforms = []
        self.particles = []

        # Create starting platforms
        start_platform = pygame.Rect(
            self.SCREEN_WIDTH / 2 - 50, self.SCREEN_HEIGHT - 30, 100, self.PLATFORM_HEIGHT
        )
        self.platforms.append(start_platform)
        
        last_platform = start_platform
        while last_platform.top > 0:
            last_platform = self._generate_next_platform(last_platform)
            self.platforms.append(last_platform)

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Tick the clock for auto-advance
        if self.auto_advance:
            self.clock.tick(self.FPS)

        # Unpack factorized action
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.1  # Survival reward

        # --- Player Input and Movement ---
        # Horizontal movement
        if movement == 3:  # Left
            self.player_pos.x -= self.PLAYER_SPEED
        if movement == 4:  # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        # Clamp player to screen horizontal bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE / 2, self.SCREEN_WIDTH - self.PLAYER_SIZE / 2)

        # Jumping
        is_jumping = (movement == 1 or space_held)
        if is_jumping and self.on_ground:
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # Sound effect placeholder: # JUMP_SOUND
            
            # Calculate aiming reward
            target_platform = self._get_next_target_platform()
            if target_platform:
                dist = abs(self.player_pos.x - target_platform.centerx)
                reward -= 0.02 * dist

        # --- Physics Update ---
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        # Update position
        self.player_pos += self.player_vel
        self.player_rect.center = self.player_pos

        # --- Platform and Collision Logic ---
        self.on_ground = False
        if self.player_vel.y > 0:  # Only check for landing if falling
            for i, plat in enumerate(self.platforms):
                if self.player_rect.colliderect(plat) and self.player_rect.bottom < plat.bottom:
                    self.player_pos.y = plat.top - self.PLAYER_SIZE / 2
                    self.player_vel.y = 0
                    self.on_ground = True
                    # Sound effect placeholder: # LAND_SOUND
                    self._create_landing_particles(self.player_rect.midbottom)
                    
                    # Reward for landing on a new, higher platform
                    if i > self.highest_platform_idx:
                        reward += 1.0
                        self.score += 1
                        self.highest_platform_idx = i
                    break
        
        # Update platforms
        for plat in self.platforms:
            plat.y -= self.platform_speed
        
        # Remove old platforms and generate new ones
        self.platforms = [p for p in self.platforms if p.bottom > 0]
        if self.platforms[-1].top > 0:
            new_plat = self._generate_next_platform(self.platforms[-1])
            self.platforms.append(new_plat)

        # --- Update Particles ---
        self._update_particles()

        # --- State and Termination ---
        self.steps += 1
        self.timer -= 1.0 / self.FPS
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.platform_speed += self.PLATFORM_SPEED_INCREASE

        terminated = False
        if self.player_rect.top > self.SCREEN_HEIGHT:  # Fell off bottom
            terminated = True
            reward = -5.0
        elif self.player_pos.y < self.GOAL_Y:  # Reached top
            terminated = True
            reward = 100.0
            self.score += 100
        elif self.timer <= 0 or self.steps >= self.MAX_STEPS:  # Time's up
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Draw platforms
        for i, plat in enumerate(self.platforms):
            color_index = (i + len(self.platforms)) % len(self.NEON_COLORS)
            color = self.NEON_COLORS[color_index]
            pygame.draw.rect(self.screen, color, plat, border_radius=3)
            # Add a subtle inner glow
            inner_rect = plat.inflate(-4, -4)
            inner_color = tuple(min(255, c + 60) for c in color)
            pygame.draw.rect(self.screen, inner_color, inner_rect, border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            radius = int(p['size'] * alpha)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), radius, p['color'] + (int(255 * alpha),))

        # Draw player
        # Glow effect
        glow_surf = pygame.Surface((self.PLAYER_SIZE * 3, self.PLAYER_SIZE * 3), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_SIZE * 1.5, self.PLAYER_SIZE * 1.5), self.PLAYER_SIZE * 1.2)
        self.screen.blit(glow_surf, (self.player_rect.centerx - self.PLAYER_SIZE * 1.5, self.player_rect.centery - self.PLAYER_SIZE * 1.5))
        # Player body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=2)

        # Draw UI
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        height_txt = f"SCORE: {self.score}"
        time_txt = f"TIME: {max(0, self.timer):.1f}"

        # Render text with shadow
        for i, text in enumerate([height_txt, time_txt]):
            pos = (12, 12 + i * 22)
            shadow_pos = (pos[0] + 2, pos[1] + 2)
            text_surf = self.font_ui.render(text, True, self.COLOR_UI_TEXT)
            shadow_surf = self.font_ui.render(text, True, self.COLOR_UI_SHADOW)
            self.screen.blit(shadow_surf, shadow_pos)
            self.screen.blit(text_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "player_y": self.player_pos.y,
        }

    def _generate_next_platform(self, last_platform):
        max_jump_height = abs(self.JUMP_STRENGTH**2 / (2 * self.GRAVITY))
        
        min_dy = 40
        max_dy = min(max_jump_height * 0.8, 120)
        
        new_y = last_platform.top - self.np_random.uniform(min_dy, max_dy)
        
        max_dx = 150
        new_x = last_platform.centerx + self.np_random.uniform(-max_dx, max_dx)
        
        min_width, max_width = 60, 120
        new_width = self.np_random.uniform(min_width, max_width)
        
        new_x = np.clip(new_x, new_width / 2, self.SCREEN_WIDTH - new_width / 2)
        
        return pygame.Rect(new_x - new_width / 2, new_y, new_width, self.PLATFORM_HEIGHT)

    def _get_next_target_platform(self):
        # Find platforms above the player
        candidate_platforms = [p for p in self.platforms if p.top < self.player_rect.top]
        if not candidate_platforms:
            return None
        # Return the lowest of those (closest vertically)
        return min(candidate_platforms, key=lambda p: p.top)

    def _create_landing_particles(self, pos):
        for _ in range(10):
            vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -1))
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'color': (255, 255, 255),
                'size': self.np_random.uniform(2, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Use a separate display for human play
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Hopper")
    
    terminated = False
    
    # Action buffer
    action = env.action_space.sample()
    action.fill(0)
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP]:
            action[0] = 1
        
        # Space
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        # Shift
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()
    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")