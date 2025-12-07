import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use Left/Right for air control. Press Space to jump. Hold Shift to fall faster."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between procedurally generated platforms, collect coins, and reach the glowing goal in this side-scrolling arcade platformer."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (20, 30, 80)
    COLOR_BG_BOTTOM = (60, 80, 150)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_PLATFORM = (180, 180, 200)
    COLOR_PLATFORM_DANGER = (255, 80, 80)
    COLOR_GOAL = (80, 255, 80)
    COLOR_GOAL_GLOW = (180, 255, 180)
    COLOR_COIN = (255, 223, 0)
    COLOR_COIN_GLOW = (255, 240, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    # Physics
    GRAVITY = 0.4
    FAST_FALL_MULTIPLIER = 2.5
    JUMP_POWER = -9.0
    AIR_CONTROL_ACCEL = 0.6
    MAX_VX = 4.0
    FRICTION = 0.9

    # Game settings
    MAX_EPISODE_STEPS = 5000
    INITIAL_LIVES = 3
    GOAL_Y_POSITION = -10000 # Relative to start
    DIFFICULTY_INTERVAL = 500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 20, bold=True)
        self.bg_surface = self._create_background()
        
        # Initialize state variables
        self.np_random = None
        self.player = {}
        self.platforms = []
        self.coins = []
        self.particles = []
        self.camera_y = 0
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.level_cleared = False
        self.max_y_reached = 0
        self.difficulty_params = {}
        
        # self.reset() is called by the wrapper/runner, not needed here usually
        # but to make the class instantiable on its own, we can call it.
        # We need to initialize the RNG first.
        if self.np_random is None:
            self.np_random = np.random.default_rng()
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # Fallback to a new generator if seed is None and it hasn't been initialized
        if self.np_random is None:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.level_cleared = False
        self.camera_y = 0
        
        self.player = {
            'x': self.SCREEN_WIDTH / 2, 'y': self.SCREEN_HEIGHT - 60,
            'vx': 0, 'vy': 0,
            'w': 20, 'h': 30,
            'on_ground': False,
            'jump_squash': 0
        }
        self.max_y_reached = self.player['y']

        self.platforms = []
        self.coins = []
        self.particles = []
        
        # Difficulty params must be initialized before generating platforms
        self.difficulty_params = {
            'min_gap_y': 80, 'max_gap_y': 140,
            'max_offset_x': 200,
            'disappear_chance': 0.1
        }

        # Initial platform
        self._add_platform(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 30, 200, 20)

        # Procedurally generate starting platforms
        last_y = self.SCREEN_HEIGHT - 30
        while last_y > -self.SCREEN_HEIGHT:
            last_y = self._generate_next_platform(last_y)

        # Goal platform
        self._add_platform(self.SCREEN_WIDTH / 2, self.GOAL_Y_POSITION, 150, 30, type='goal')
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01 # Time penalty
        
        # --- Update game logic based on action ---
        self._handle_input(action)
        player_reward = self._update_player()
        entities_reward = self._update_entities()
        reward += player_reward + entities_reward

        self.steps += 1
        
        # --- Check for termination ---
        terminated = self.game_over or self.steps >= self.MAX_EPISODE_STEPS
        truncated = False # This environment does not truncate based on time limit

        if self.game_over and not self.level_cleared:
            reward -= 5.0 # Penalty for falling and losing
        if self.level_cleared:
            reward += 100.0 # Huge reward for winning
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        # Horizontal air control
        if not self.player['on_ground']:
            if movement == 3: # Left
                self.player['vx'] -= self.AIR_CONTROL_ACCEL
            elif movement == 4: # Right
                self.player['vx'] += self.AIR_CONTROL_ACCEL

        # Jumping
        if space_held and self.player['on_ground']:
            self.player['vy'] = self.JUMP_POWER
            self.player['on_ground'] = False
            self.player['jump_squash'] = 10 # Start jump animation
            # sfx: jump

        # Fast falling
        self.player['fast_fall'] = shift_held and self.player['vy'] > 0

    def _update_player(self):
        # Apply physics
        gravity = self.GRAVITY * self.FAST_FALL_MULTIPLIER if self.player.get('fast_fall') else self.GRAVITY
        self.player['vy'] += gravity
        self.player['vx'] *= self.FRICTION
        self.player['vx'] = np.clip(self.player['vx'], -self.MAX_VX, self.MAX_VX)
        
        self.player['x'] += self.player['vx']
        self.player['y'] += self.player['vy']

        # Wall bouncing
        if self.player['x'] < self.player['w'] / 2:
            self.player['x'] = self.player['w'] / 2
            self.player['vx'] *= -0.5
        if self.player['x'] > self.SCREEN_WIDTH - self.player['w'] / 2:
            self.player['x'] = self.SCREEN_WIDTH - self.player['w'] / 2
            self.player['vx'] *= -0.5

        # Check for falling out of bounds
        if self.player['y'] > self.camera_y + self.SCREEN_HEIGHT + 50:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
            else:
                # Respawn at the highest platform reached
                highest_plat = min(self.platforms, key=lambda p: abs(p['y'] - self.max_y_reached), default=None)
                if highest_plat:
                    self.player['x'], self.player['y'] = highest_plat['x'], highest_plat['y'] - 50
                else: # Failsafe if no platforms exist
                    self.player['x'], self.player['y'] = self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT - 60
                self.player['vx'], self.player['vy'] = 0, 0
                return -5.0 # Return penalty for respawn
        return 0

    def _update_entities(self):
        reward = 0
        
        # Update camera
        target_camera_y = self.camera_y
        if self.player['y'] < self.camera_y + self.SCREEN_HEIGHT / 3:
            target_camera_y = self.player['y'] - self.SCREEN_HEIGHT / 3
        self.camera_y += (target_camera_y - self.camera_y) * 0.1

        # Player-platform collision
        self.player['on_ground'] = False
        player_rect = pygame.Rect(self.player['x'] - self.player['w']/2, self.player['y'] - self.player['h'], self.player['w'], self.player['h'])
        
        for plat in self.platforms:
            plat_rect = pygame.Rect(plat['x'] - plat['w']/2, plat['y'] - plat['h']/2, plat['w'], plat['h'])
            if player_rect.colliderect(plat_rect) and self.player['vy'] > 0:
                # Check if player was above the platform in the previous frame
                prev_player_bottom = player_rect.bottom - self.player['vy']
                if prev_player_bottom <= plat_rect.top:
                    self.player['y'] = plat_rect.top
                    self.player['vy'] = 0
                    self.player['on_ground'] = True
                    self.player['jump_squash'] = -10 # Start landing animation
                    if plat.get('disappear_timer') is not None:
                        plat['disappear_timer'] = max(1, plat['disappear_timer'] - 15) # Speed up disappearance on land
                    
                    self._create_particles(self.player['x'], self.player['y'], 5, (200,200,200))

                    if plat['y'] < self.max_y_reached:
                        reward += (self.max_y_reached - plat['y']) / 50.0 # Reward for vertical progress
                        self.max_y_reached = plat['y']
                    
                    if plat.get('type') == 'goal':
                        self.level_cleared = True
                        self.game_over = True # End game on win
                    break

        # Coin collection
        player_center_rect = pygame.Rect(self.player['x'] - 5, self.player['y'] - self.player['h']/2 - 5, 10, 10)
        for coin in self.coins[:]:
            coin_rect = pygame.Rect(coin['x'] - 10, coin['y'] - 10, 20, 20)
            if player_center_rect.colliderect(coin_rect):
                self.coins.remove(coin)
                self.score += 10
                reward += 1.0
                self._create_particles(coin['x'], coin['y'], 10, self.COLOR_COIN)

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Particle gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
        
        # Update platforms (disappearing)
        for p in self.platforms[:]:
            if p.get('disappear_timer') is not None:
                p['disappear_timer'] -= 1
                if p['disappear_timer'] <= 0:
                    self._create_particles(p['x'], p['y'], 15, self.COLOR_PLATFORM_DANGER)
                    self.platforms.remove(p)

        # Generate new platforms if needed
        if len(self.platforms) > 0:
            highest_platform_y = min(p['y'] for p in self.platforms)
            if highest_platform_y > self.camera_y - self.SCREEN_HEIGHT:
                self._generate_next_platform(highest_platform_y)
        
        # Prune old entities
        self.platforms = [p for p in self.platforms if p['y'] < self.camera_y + self.SCREEN_HEIGHT + 50]
        self.coins = [c for c in self.coins if c['y'] < self.camera_y + self.SCREEN_HEIGHT + 50]
        
        # Update difficulty
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.difficulty_params['min_gap_y'] = max(60, self.difficulty_params['min_gap_y'] - 2)
            self.difficulty_params['max_gap_y'] = max(100, self.difficulty_params['max_gap_y'] - 2)
            self.difficulty_params['max_offset_x'] = min(300, self.difficulty_params['max_offset_x'] + 5)
            self.difficulty_params['disappear_chance'] = min(0.4, self.difficulty_params['disappear_chance'] + 0.02)

        return reward

    def _get_observation(self):
        # Clear screen with background
        self.screen.blit(self.bg_surface, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_y = int(self.camera_y)

        # Render platforms
        for plat in self.platforms:
            px, py = int(plat['x']), int(plat['y'] - cam_y)
            pw, ph = int(plat['w']), int(plat['h'])
            rect = (px - pw//2, py - ph//2, pw, ph)
            
            color = self.COLOR_PLATFORM
            if plat.get('type') == 'goal':
                self._draw_glowing_rect(rect, self.COLOR_GOAL, self.COLOR_GOAL_GLOW)
            elif plat.get('disappear_timer') is not None:
                timer = plat['disappear_timer']
                if timer < 120 and (timer // 10) % 2 == 0:
                    color = self.COLOR_PLATFORM_DANGER
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
            else:
                 pygame.draw.rect(self.screen, color, rect, border_radius=5)

        # Render coins
        for coin in self.coins:
            cx, cy = int(coin['x']), int(coin['y'] - cam_y)
            # Spinning animation
            anim_width = int(8 * abs(math.sin(self.steps * 0.1 + coin['x'])))
            self._draw_glowing_ellipse((cx, cy), (anim_width, 10), self.COLOR_COIN, self.COLOR_COIN_GLOW)

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
            color = p['color'] + (alpha,)
            pos = (int(p['pos'][0]), int(p['pos'][1] - cam_y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['lifespan'] * 0.2), color)

        # Render player
        if self.lives > 0:
            # Jump/land squash and stretch animation
            if self.player['jump_squash'] != 0:
                squash = self.player['jump_squash'] / 10.0
                w_mod = -squash * 5
                h_mod = squash * 8
                self.player['jump_squash'] -= 1 if self.player['jump_squash'] > 0 else -1
            else:
                w_mod, h_mod = 0, 0
            
            pw = int(self.player['w'] + w_mod)
            ph = int(self.player['h'] + h_mod)
            px = int(self.player['x'])
            py = int(self.player['y'] - cam_y)
            player_rect = pygame.Rect(px - pw//2, py - ph, pw, ph)
            
            # Glow effect
            glow_radius = int(ph * 0.8)
            glow_center = (px, py - ph // 2)
            for i in range(glow_radius // 2, 0, -2):
                alpha = 40 * (1 - (i / (glow_radius // 2)))
                pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], i, (*self.COLOR_PLAYER_GLOW, int(alpha)))
            
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=7)

    def _render_ui(self):
        # Score
        self._draw_text(f"SCORE: {self.score}", (20, 10), self.font_small)
        # Lives
        self._draw_text(f"LIVES: {self.lives}", (self.SCREEN_WIDTH - 120, 10), self.font_small)

        # Game Over / Level Cleared
        if self.game_over and not self.level_cleared:
            self._draw_text("GAME OVER", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 30), self.font_large, center=True)
        elif self.level_cleared:
            self._draw_text("LEVEL CLEARED!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 30), self.font_large, center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "level_cleared": self.level_cleared
        }
        
    # --- Helper Functions ---

    def _add_platform(self, x, y, w, h, type='normal'):
        plat = {'x': x, 'y': y, 'w': w, 'h': h, 'type': type}
        if type == 'disappearing':
            plat['disappear_timer'] = self.np_random.integers(150, 400)
        self.platforms.append(plat)
        # Potentially add a coin on the platform
        if type == 'normal' and self.np_random.random() < 0.4:
            self.coins.append({'x': x, 'y': y - 20})

    def _generate_next_platform(self, last_y):
        gap_y = self.np_random.integers(self.difficulty_params['min_gap_y'], self.difficulty_params['max_gap_y'])
        new_y = last_y - gap_y
        
        offset_x = self.np_random.integers(-self.difficulty_params['max_offset_x'], self.difficulty_params['max_offset_x'] + 1)
        new_x = np.clip(self.SCREEN_WIDTH / 2 + offset_x, 80, self.SCREEN_WIDTH - 80)
        
        new_w = self.np_random.integers(70, 130)
        new_h = 20
        
        plat_type = 'normal'
        if self.np_random.random() < self.difficulty_params['disappear_chance']:
            plat_type = 'disappearing'
        
        self._add_platform(new_x, new_y, new_w, new_h, type=plat_type)
        return new_y

    def _create_particles(self, x, y, count, color):
        for _ in range(count):
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': [x, y],
                'vel': [(self.np_random.random() - 0.5) * 4, (self.np_random.random() - 0.8) * 4],
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color
            })

    def _create_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = [
                self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp
                for i in range(3)
            ]
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _draw_glowing_rect(self, rect, color, glow_color):
        px, py, pw, ph = rect
        for i in range(10, 0, -2):
            alpha = 80 * (1 - (i / 10))
            glow_rect = (px - i, py - i, pw + 2 * i, ph + 2 * i)
            s = pygame.Surface((pw + 2 * i, ph + 2 * i), pygame.SRCALPHA)
            pygame.draw.rect(s, (*glow_color, int(alpha)), (0, 0, *s.get_size()), border_radius=10)
            self.screen.blit(s, (px-i, py-i))
        pygame.draw.rect(self.screen, color, rect, border_radius=5)

    def _draw_glowing_ellipse(self, center, size, color, glow_color):
        cx, cy = center
        sw, sh = size
        rect = (cx - sw, cy - sh, sw * 2, sh * 2)
        
        for i in range(5, 0, -1):
            alpha = 100 * (1 - (i/5))
            glow_size = (sw + i, sh + i)
            s = pygame.Surface((glow_size[0]*2, glow_size[1]*2), pygame.SRCALPHA)
            pygame.draw.ellipse(s, (*glow_color, int(alpha)), (0,0, *s.get_size()))
            self.screen.blit(s, (cx - glow_size[0], cy - glow_size[1]))

        pygame.draw.ellipse(self.screen, color, rect)

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    
    # Unset the dummy video driver if you want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Platformer")
    
    done = False
    total_reward = 0
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No movement by default
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        # Space (Jump)
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift (Fast Fall)
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Render ---
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60) # Run at 60 FPS
        
        if done:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before closing
            pygame.time.wait(2000)
            
    env.close()