
# Generated: 2025-08-28T06:08:47.502586
# Source Brief: brief_05802.md
# Brief Index: 5802

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game objects
class Player:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.vx = 0
        self.vy = 0
        self.state = 'idle'  # 'idle' or 'jumping'
        self.on_platform = None
        self.trail = []

    def update(self, gravity):
        if self.state == 'jumping':
            self.vy += gravity
            self.rect.x += self.vx
            self.rect.y += self.vy

        # Update trail
        self.trail.append(self.rect.copy())
        if len(self.trail) > 5:
            self.trail.pop(0)

    def jump(self, dx, dy):
        self.state = 'jumping'
        self.vx = dx
        self.vy = dy
        self.on_platform = None
        # Sound placeholder: # pygame.mixer.Sound('jump.wav').play()

    def land(self, platform):
        self.state = 'idle'
        self.vx = 0
        self.vy = 0
        self.rect.bottom = platform.rect.top
        self.on_platform = platform

class Platform:
    def __init__(self, rect, p_type='safe'):
        self.rect = rect
        self.type = p_type  # 'safe' or 'risky'
        self.flash_timer = 0
        self.disappear_timer = -1 # -1 means permanent, >0 is a countdown

    def update(self):
        if self.flash_timer > 0:
            self.flash_timer -= 1
        if self.disappear_timer > 0:
            self.disappear_timer -= 1
        
    def is_active(self):
        return self.disappear_timer != 0

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-3, 1)
        self.lifespan = random.randint(15, 30)
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1 # particle gravity
        self.lifespan -= 1


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to jump up, ←→ to jump diagonally. ↓ to drop through your current platform."
    )

    game_description = (
        "Leap across procedurally generated platforms to reach the top. Risky red platforms give more points but disappear. Reach the top before time runs out!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.GAME_DURATION_SECONDS = 30
        
        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLATFORM_SAFE = (0, 255, 128)
        self.COLOR_PLATFORM_RISKY = (255, 50, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_PARTICLE = (255, 255, 255)

        # Physics
        self.GRAVITY = 0.5
        self.JUMP_VELOCITY_Y = -10
        self.JUMP_VELOCITY_X = 5
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Game state variables
        self.player = None
        self.platforms = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.time_left = 0
        self.game_over = False
        self.total_scroll = 0
        self.goal_y = -5000 # Target height to reach
        
        # Difficulty scaling
        self.base_platform_width = 100
        self.min_platform_width = 30
        self.red_platform_chance = 0.1
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.time_left = self.GAME_DURATION_SECONDS * self.FPS
        self.game_over = False
        self.total_scroll = 0
        
        self.red_platform_chance = 0.1
        self.base_platform_width = 100

        self.platforms = []
        self.particles = []

        # Create initial platforms
        start_platform = Platform(pygame.Rect(self.WIDTH // 2 - 50, self.HEIGHT - 50, 100, 20), 'safe')
        self.platforms.append(start_platform)
        self.player = Player(start_platform.rect.centerx - 10, start_platform.rect.top - 20, 20, 20)
        self.player.land(start_platform)

        # Generate starting screen of platforms
        for y in range(self.HEIGHT - 150, 0, -80):
             self._generate_platforms_at(y)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = self.game_over

        if not terminated:
            movement = action[0]
            
            # --- 1. Handle Input & Calculate Action-based Rewards ---
            reward += self._handle_input(movement)
            
            # --- 2. Update Game State ---
            self.player.update(self.GRAVITY)
            
            # Remove inactive objects
            self.platforms = [p for p in self.platforms if p.is_active()]
            self.particles = [p for p in self.particles if p.lifespan > 0]
            for p in self.platforms: p.update()
            for p in self.particles: p.update()
            
            # --- 3. Collision Detection ---
            landed_reward = self._check_collisions()
            reward += landed_reward

            # --- 4. World Scrolling & Progression ---
            scroll_reward = self._scroll_world()
            reward += scroll_reward
            self._generate_new_platforms()
            self._update_difficulty()
            
            # --- 5. Update Timers & Check Termination ---
            self.time_left -= 1
            self.steps += 1
            
            # Check for falling off screen
            if self.player.rect.top > self.HEIGHT:
                self.lives -= 1
                # Sound placeholder: # pygame.mixer.Sound('fall.wav').play()
                if self.lives > 0:
                    self._respawn_player()
                else:
                    self.game_over = True
                    reward -= 100 # Lose all lives penalty

            # Check other termination conditions
            if self.time_left <= 0:
                self.game_over = True
                reward -= 50 # Time out penalty
            
            if self.total_scroll >= abs(self.goal_y):
                self.game_over = True
                reward += 100 # Win bonus
                # Sound placeholder: # pygame.mixer.Sound('win.wav').play()

            if self.steps >= self.MAX_STEPS:
                self.game_over = True

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        reward = 0
        if self.player.state == 'idle':
            if movement == 0: # No-op
                reward -= 0.05
            elif movement == 1: # Jump Up
                self.player.jump(0, self.JUMP_VELOCITY_Y)
            elif movement == 2: # Drop Down
                # Find platform to drop to
                can_drop = False
                for p in self.platforms:
                    if p != self.player.on_platform and p.rect.top > self.player.rect.bottom and p.rect.colliderect(self.player.rect.x, self.player.rect.y + 1, self.player.rect.width, self.HEIGHT):
                        can_drop = True
                        break
                if can_drop:
                    self.player.rect.y += 1 # Nudge to fall through
                    self.player.jump(0, 0)
            elif movement == 3: # Jump Left
                self.player.jump(-self.JUMP_VELOCITY_X, self.JUMP_VELOCITY_Y * 0.8)
                reward -= 0.02 # Sideways movement penalty
            elif movement == 4: # Jump Right
                self.player.jump(self.JUMP_VELOCITY_X, self.JUMP_VELOCITY_Y * 0.8)
                reward -= 0.02 # Sideways movement penalty
        return reward

    def _check_collisions(self):
        reward = 0
        if self.player.vy > 0: # Only check for landing if falling
            for p in self.platforms:
                if p.rect.colliderect(self.player.rect) and abs(self.player.rect.bottom - p.rect.top) < self.player.vy + 1:
                    self.player.land(p)
                    p.flash_timer = 10
                    # Sound placeholder: # pygame.mixer.Sound('land.wav').play()
                    
                    # Create landing particles
                    for _ in range(10):
                        self.particles.append(Particle(self.player.rect.centerx, self.player.rect.bottom, self.COLOR_PARTICLE))

                    # Calculate landing reward
                    if p.type == 'safe':
                        reward += 2
                    elif p.type == 'risky':
                        reward += 5
                        p.disappear_timer = self.FPS // 2 # Disappears in 0.5s
                    
                    # Penalty for imprecise landing
                    intersection = self.player.rect.clip(p.rect)
                    if intersection.width < self.player.rect.width * 0.2:
                        reward -= 1
                    
                    break
        return reward

    def _scroll_world(self):
        reward = 0
        scroll_threshold = self.HEIGHT / 3
        if self.player.rect.y < scroll_threshold:
            scroll_dist = scroll_threshold - self.player.rect.y
            self.player.rect.y += scroll_dist
            self.total_scroll += scroll_dist
            self.score = int(self.total_scroll)
            reward += scroll_dist * 0.01 # Reward for upward progress (scaled from +0.1 per step)

            for p in self.platforms:
                p.rect.y += scroll_dist
            for part in self.particles:
                part.y += scroll_dist
        return reward

    def _generate_new_platforms(self):
        # Remove platforms that are off-screen
        self.platforms = [p for p in self.platforms if p.rect.top < self.HEIGHT]
        
        # Find the highest platform to generate new ones above it
        highest_y = self.HEIGHT
        if self.platforms:
            highest_y = min(p.rect.y for p in self.platforms)
        
        # Generate new platforms if space is available at the top
        while highest_y > -100:
            new_y = highest_y - random.randint(50, 100)
            self._generate_platforms_at(new_y)
            highest_y = new_y

    def _generate_platforms_at(self, y_pos):
        x_pos = random.randint(0, self.WIDTH - int(self.base_platform_width))
        width = max(self.min_platform_width, int(self.base_platform_width - random.randint(0, 20)))
        rect = pygame.Rect(x_pos, y_pos, width, 20)
        
        # Avoid overlap with existing platforms
        is_overlapping = any(rect.colliderect(p.rect) for p in self.platforms)
        if not is_overlapping:
            p_type = 'risky' if random.random() < self.red_platform_chance else 'safe'
            self.platforms.append(Platform(rect, p_type))

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 100 == 0:
            self.red_platform_chance = min(0.75, self.red_platform_chance + 0.02)
        if self.steps > 0 and self.steps % 200 == 0:
            self.base_platform_width = max(self.min_platform_width, self.base_platform_width - 1)

    def _respawn_player(self):
        # Find a safe platform to respawn on, or create one
        safe_platforms = [p for p in self.platforms if p.rect.top > self.HEIGHT/2 and p.rect.bottom < self.HEIGHT and p.is_active()]
        if safe_platforms:
            respawn_platform = random.choice(safe_platforms)
        else:
            respawn_platform = Platform(pygame.Rect(self.WIDTH // 2 - 50, self.HEIGHT - 100, 100, 20), 'safe')
            self.platforms.append(respawn_platform)
        
        self.player = Player(respawn_platform.rect.centerx - 10, respawn_platform.rect.top - 20, 20, 20)
        self.player.land(respawn_platform)
        self.player.trail.clear()

    def _get_observation(self):
        self._render_background()
        self._render_platforms()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.HEIGHT):
            color = [
                int(self.COLOR_BG_TOP[i] + (self.COLOR_BG_BOTTOM[i] - self.COLOR_BG_TOP[i]) * (y / self.HEIGHT))
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_platforms(self):
        for p in self.platforms:
            base_color = self.COLOR_PLATFORM_SAFE if p.type == 'safe' else self.COLOR_PLATFORM_RISKY
            if p.flash_timer > 0:
                # Interpolate to white for a flash effect
                flash_alpha = p.flash_timer / 10.0
                color = [int(base_color[i] + (255 - base_color[i]) * flash_alpha) for i in range(3)]
            else:
                color = base_color
            
            if p.disappear_timer > 0:
                alpha = int(255 * (p.disappear_timer / (self.FPS / 2)))
                temp_surf = pygame.Surface(p.rect.size, pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, color + (alpha,), (0,0, *p.rect.size), border_radius=3)
                self.screen.blit(temp_surf, p.rect.topleft)
            else:
                 pygame.draw.rect(self.screen, color, p.rect, border_radius=3)
                 pygame.draw.rect(self.screen, tuple(max(0, c-40) for c in color), p.rect, width=2, border_radius=3)


    def _render_player(self):
        # Draw trail
        for i, rect in enumerate(self.player.trail):
            alpha = int(255 * (i / len(self.player.trail)) * 0.5)
            trail_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            trail_surf.fill((*self.COLOR_PLAYER, alpha))
            self.screen.blit(trail_surf, rect.topleft)

        # Draw player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player.rect)
        pygame.draw.rect(self.screen, (255,255,255), self.player.rect, 1)

    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p.lifespan / 30)
            color = (*p.color, alpha)
            temp_surf = pygame.Surface((3,3), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (1, 1), 1)
            self.screen.blit(temp_surf, (int(p.x), int(p.y)))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT - 40))

        # Time
        time_str = f"TIME: {max(0, self.time_left // self.FPS):02d}"
        time_text = self.font_small.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Lives (hearts)
        for i in range(self.lives):
            self._draw_heart(15 + i * 25, 20)
            
        # Goal progress
        progress = min(1.0, self.total_scroll / abs(self.goal_y))
        bar_width = self.WIDTH - 40
        pygame.draw.rect(self.screen, (50,50,50), (20, self.HEIGHT-10, bar_width, 5))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (20, self.HEIGHT-10, bar_width * progress, 5))


    def _draw_heart(self, x, y):
        # A simple heart shape using gfxdraw
        c = (255, 80, 80)
        pygame.gfxdraw.aacircle(self.screen, x - 5, y - 5, 5, c)
        pygame.gfxdraw.filled_circle(self.screen, x - 5, y - 5, 5, c)
        pygame.gfxdraw.aacircle(self.screen, x + 5, y - 5, 5, c)
        pygame.gfxdraw.filled_circle(self.screen, x + 5, y - 5, 5, c)
        points = [(x - 10, y - 4), (x, y + 6), (x + 10, y - 4)]
        pygame.gfxdraw.aapolygon(self.screen, points, c)
        pygame.gfxdraw.filled_polygon(self.screen, points, c)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "time_left": self.time_left // self.FPS,
        }
        
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


if __name__ == '__main__':
    # To play the game manually
    import os
    # This is required for environments where a display is not available.
    # If you are running on a server, you may need to install xvfb.
    # `pip install pyvirtualdisplay` and `apt-get install xvfb`
    # then add `from pyvirtualdisplay import Display; Display().start()`
    if os.name != "nt": # Not windows
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    
    env = GameEnv(render_mode="rgb_array")
    
    # This part is for human play, and requires a display.
    # If you are on a server, comment out this block.
    try:
        if os.name == "nt":
             os.environ['SDL_VIDEODRIVER'] = 'windows'
        else:
             os.environ['SDL_VIDEODRIVER'] = 'x11'

        obs, info = env.reset()
        
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Platformer Environment")
        
        running = True
        total_reward = 0
        
        # Mapping keys to actions
        key_map = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }

        while running:
            movement_action = 0 # Default to no-op
            
            keys = pygame.key.get_pressed()
            for key, action_val in key_map.items():
                if keys[key]:
                    movement_action = action_val
                    break

            # In this game, space and shift are not used, so they are always 0
            action = [movement_action, 0, 0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation from the environment to the display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                total_reward = 0
                obs, info = env.reset()
                # Add a small delay before restarting
                pygame.time.wait(2000)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

    except pygame.error as e:
        print("\nCould not create display. This is expected on a server.")
        print("The environment is still valid for training, but cannot be rendered for human play.")
        print("Running validation...")
        env.validate_implementation()

    finally:
        env.close()