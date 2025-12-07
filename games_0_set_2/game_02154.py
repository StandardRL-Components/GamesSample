
# Generated: 2025-08-27T19:25:29.036195
# Source Brief: brief_02154.md
# Brief Index: 2154

        
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
        "Controls: Use arrow keys to jump. Hold Shift for a long left jump, or Space for a long right jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling arcade game where you hop between platforms, collecting coins to reach the level's end."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    LEVEL_END_X = 10000
    MAX_STEPS = 5000
    
    # Colors
    COLOR_BG = (15, 15, 40)
    COLOR_PLAYER = (255, 60, 60)
    COLOR_PLAYER_GLOW = (255, 150, 150, 50)
    COLOR_PLATFORM = (180, 180, 190)
    COLOR_PLATFORM_UNSTABLE = (200, 150, 120)
    COLOR_COIN = (255, 223, 0)
    COLOR_TEXT = (240, 240, 240)
    
    # Physics
    GRAVITY = 0.4
    JUMP_VEL_NORMAL = -8
    JUMP_VEL_LONG = -10
    JUMP_X_VEL_NORMAL = 4
    JUMP_X_VEL_LONG = 7
    MAX_FALL_SPEED = 10
    PLAYER_FRICTION = 0.9

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_size = pygame.Vector2(20, 30)
        self.player_squash = 1.0 # For animation
        self.is_grounded = False
        self.last_safe_pos = pygame.Vector2(0, 0)
        
        self.platforms = []
        self.coins = []
        self.particles = []
        self.stars = []
        
        self.camera_offset_x = 0
        self.rng = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 4, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = False
        
        self.platform_scroll_speed = 1.0
        self.unstable_platform_prob = 0.1
        
        self.platforms.clear()
        self.coins.clear()
        self.particles.clear()
        self._initialize_level()
        self._generate_stars()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.01 # Liveness reward
        self.steps += 1

        self._handle_input(action)
        self._update_player()
        self._update_world()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        self._update_difficulty()

        terminated = self._check_termination()
        if terminated:
            if self.player_pos.y > self.SCREEN_HEIGHT + 50:
                self.lives -= 1
                reward -= 50
                if self.lives <= 0:
                    self.game_over = True
                    reward -= 100 # Final game over penalty
                else:
                    self._respawn_player()
                    terminated = False # Continue playing
            
            elif self.player_pos.x >= self.LEVEL_END_X:
                reward += 100 # Victory reward
                self.game_over = True

            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _initialize_level(self):
        # Create starting platform
        start_plat = self._create_platform(pygame.Rect(self.SCREEN_WIDTH / 4 - 50, self.SCREEN_HEIGHT / 2 + 50, 150, 20), False)
        self.platforms.append(start_plat)
        self.player_pos = pygame.Vector2(start_plat['rect'].centerx, start_plat['rect'].top - self.player_size.y)
        self.last_safe_pos = self.player_pos.copy()

        # Generate initial platforms
        while len(self.platforms) < 15:
            self._generate_new_platform()

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': pygame.Vector2(self.rng.integers(0, self.SCREEN_WIDTH), self.rng.integers(0, self.SCREEN_HEIGHT)),
                'depth': self.rng.uniform(0.1, 0.6) # Determines parallax speed
            })

    def _handle_input(self, action):
        if not self.is_grounded:
            return

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        jumped = False
        if shift_held:
            # Long jump left
            self.player_vel.y = self.JUMP_VEL_LONG
            self.player_vel.x = -self.JUMP_X_VEL_LONG
            jumped = True
        elif space_held:
            # Long jump right
            self.player_vel.y = self.JUMP_VEL_LONG
            self.player_vel.x = self.JUMP_X_VEL_LONG
            jumped = True
        elif movement == 1: # Up
            self.player_vel.y = self.JUMP_VEL_NORMAL
            self.player_vel.x = 0
            jumped = True
        elif movement == 3: # Left
            self.player_vel.y = self.JUMP_VEL_NORMAL
            self.player_vel.x = -self.JUMP_X_VEL_NORMAL
            jumped = True
        elif movement == 4: # Right
            self.player_vel.y = self.JUMP_VEL_NORMAL
            self.player_vel.x = self.JUMP_X_VEL_NORMAL
            jumped = True

        if jumped:
            self.is_grounded = False
            self.player_squash = 1.5 # Stretch for jump
            # sfx: jump

    def _update_player(self):
        # Apply gravity
        if not self.is_grounded:
            self.player_vel.y += self.GRAVITY
            self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)
        
        # Apply friction
        self.player_vel.x *= self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1:
            self.player_vel.x = 0

        # Update position
        self.player_pos += self.player_vel
        
        # Update squash animation
        self.player_squash += (1.0 - self.player_squash) * 0.2

    def _update_world(self):
        # Update camera to follow player
        target_camera_x = self.player_pos.x - self.SCREEN_WIDTH / 3
        self.camera_offset_x += (target_camera_x - self.camera_offset_x) * 0.1

        # Update and despawn platforms/coins
        self.platforms = [p for p in self.platforms if p['rect'].right > self.camera_offset_x]
        self.coins = [c for c in self.coins if c['rect'].right > self.camera_offset_x]
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        # Generate new platforms if needed
        if len(self.platforms) < 15:
            self._generate_new_platform()
            
        # Update unstable platforms
        for p in self.platforms:
            if p['unstable'] and p['timer'] > 0:
                p['timer'] -= 1
                if p['timer'] == 0:
                    self._create_debris(p['rect'])
                    # sfx: crumble
        self.platforms = [p for p in self.platforms if not (p['unstable'] and p['timer'] == 0)]
        
        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += 0.1 # Particle gravity
            p['life'] -= 1

    def _generate_new_platform(self):
        last_plat = self.platforms[-1]['rect']
        
        gap_x = self.rng.integers(80, 200)
        gap_y = self.rng.integers(-80, 80)
        
        width = self.rng.integers(80, 200)
        height = 20
        
        new_x = last_plat.right + gap_x
        new_y = np.clip(last_plat.y + gap_y, 100, self.SCREEN_HEIGHT - 50)
        
        is_unstable = self.rng.random() < self.unstable_platform_prob
        new_plat = self._create_platform(pygame.Rect(new_x, new_y, width, height), is_unstable)
        self.platforms.append(new_plat)

        # Maybe add a coin
        if self.rng.random() < 0.6:
            is_risky = self.rng.random() < 0.2
            coin_y_offset = -40 if not is_risky else self.rng.integers(-80, -50)
            coin_pos = pygame.Vector2(new_plat['rect'].centerx, new_plat['rect'].top + coin_y_offset)
            self.coins.append(self._create_coin(coin_pos, is_risky))

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        
        # Platform collisions
        self.is_grounded = False
        if self.player_vel.y > 0:
            for p in self.platforms:
                if player_rect.colliderect(p['rect']) and player_rect.bottom < p['rect'].centery:
                    self.player_pos.y = p['rect'].top - self.player_size.y
                    self.player_vel.y = 0
                    self.is_grounded = True
                    self.player_squash = 0.5 # Squash on land
                    # sfx: land
                    
                    if not p['unstable']:
                        self.last_safe_pos = self.player_pos.copy()
                    else:
                        if p['timer'] == -1: # First landing
                           p['timer'] = 60 # 2 seconds at 30fps
                           reward -= 1
                    break
        
        # Coin collisions
        for coin in self.coins[:]:
            if player_rect.colliderect(coin['rect']):
                reward += 5 if coin['risky'] else 1
                self.score += 5 if coin['risky'] else 1
                self._create_coin_particles(coin['rect'].center)
                self.coins.remove(coin)
                # sfx: coin_get
                
        return reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 500 == 0:
            self.unstable_platform_prob = min(0.5, self.unstable_platform_prob + 0.02)
        # Scroll speed is implicitly increased by player moving right

    def _check_termination(self):
        return (
            self.player_pos.y > self.SCREEN_HEIGHT + 50 or
            self.player_pos.x >= self.LEVEL_END_X or
            self.steps >= self.MAX_STEPS
        )

    def _respawn_player(self):
        self.player_pos = self.last_safe_pos.copy()
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = True
        # sfx: lose_life

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_objects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.stars:
            screen_x = (star['pos'].x - self.camera_offset_x * star['depth']) % self.SCREEN_WIDTH
            screen_y = star['pos'].y
            brightness = int(100 + 155 * star['depth'])
            pygame.draw.circle(self.screen, (brightness, brightness, brightness), (int(screen_x), int(screen_y)), 1)
        
        # Render end goal flag if close
        end_flag_screen_x = self.LEVEL_END_X - self.camera_offset_x
        if end_flag_screen_x < self.SCREEN_WIDTH + 50:
            pygame.draw.line(self.screen, self.COLOR_TEXT, (end_flag_screen_x, 0), (end_flag_screen_x, self.SCREEN_HEIGHT), 3)
            for i in range(10):
                color = self.COLOR_TEXT if i % 2 == 0 else self.COLOR_BG
                pygame.draw.rect(self.screen, color, (end_flag_screen_x, i*40, 40, 40))

    def _render_game_objects(self):
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            p['color'].a = alpha
            s = pygame.Surface((p['size'], p['size']), pygame.SRCALPHA)
            s.fill(p['color'])
            self.screen.blit(s, (p['pos'].x - self.camera_offset_x, p['pos'].y))

        # Platforms
        for p in self.platforms:
            screen_rect = p['rect'].copy()
            screen_rect.x -= self.camera_offset_x
            color = self.COLOR_PLATFORM_UNSTABLE if p['unstable'] else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, screen_rect, border_radius=3)
            if p['unstable']:
                if p['timer'] != -1 and p['timer'] % 10 < 5:
                    pygame.draw.rect(self.screen, (255, 0, 0), screen_rect, 2, border_radius=3)
                else: # Draw cracks
                    pygame.draw.line(self.screen, self.COLOR_BG, (screen_rect.left+10, screen_rect.top+5), (screen_rect.left+20, screen_rect.bottom-5), 2)
                    pygame.draw.line(self.screen, self.COLOR_BG, (screen_rect.centerx, screen_rect.top+5), (screen_rect.centerx-10, screen_rect.bottom-5), 2)

        # Coins
        for c in self.coins:
            screen_rect = c['rect'].copy()
            screen_rect.x -= self.camera_offset_x
            spin_phase = (self.steps + c['rect'].x) % 60 / 60.0
            width = int(c['rect'].width * abs(math.cos(spin_phase * math.pi * 2)))
            
            if width > 1:
                display_rect = pygame.Rect(0, 0, width, c['rect'].height)
                display_rect.center = screen_rect.center
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, display_rect)
                pygame.draw.ellipse(self.screen, tuple(min(255, x+50) for x in self.COLOR_COIN), display_rect, 2)

        # Player
        psx, psy = self.player_size.x, self.player_size.y
        squashed_w = psx / self.player_squash
        squashed_h = psy * self.player_squash
        player_screen_pos = self.player_pos - pygame.Vector2(self.camera_offset_x, 0)
        player_rect = pygame.Rect(0, 0, squashed_w, squashed_h)
        player_rect.bottomleft = player_screen_pos + pygame.Vector2(0, self.player_size.y)
        
        # Glow
        glow_radius = int(max(player_rect.width, player_rect.height) * 0.8)
        glow_center = (int(player_rect.centerx), int(player_rect.centery))
        temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
        self.screen.blit(temp_surf, (glow_center[0] - glow_radius, glow_center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=5)
        # "Helmet"
        helmet_rect = pygame.Rect(0,0, player_rect.width * 0.7, player_rect.height * 0.5)
        helmet_rect.center = (player_rect.centerx, player_rect.centery - player_rect.height*0.1)
        pygame.draw.rect(self.screen, (200, 220, 255), helmet_rect, border_radius=4)


    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        if self.game_over:
            msg = "LEVEL COMPLETE!" if self.player_pos.x >= self.LEVEL_END_X else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20,20))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "player_x": self.player_pos.x
        }

    # --- Factory Methods for Game Objects ---
    def _create_platform(self, rect, unstable):
        return {'rect': rect, 'unstable': unstable, 'timer': -1}

    def _create_coin(self, pos, risky):
        size = 15
        return {'rect': pygame.Rect(pos.x - size/2, pos.y - size/2, size, size), 'risky': risky}

    def _create_debris(self, rect):
        for _ in range(10):
            self.particles.append({
                'pos': pygame.Vector2(rect.center),
                'vel': pygame.Vector2(self.rng.uniform(-3, 3), self.rng.uniform(-3, 0)),
                'life': self.rng.integers(20, 40),
                'max_life': 40,
                'color': pygame.Color(*self.COLOR_PLATFORM_UNSTABLE, 255),
                'size': self.rng.integers(3, 7)
            })

    def _create_coin_particles(self, pos):
        for _ in range(15):
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(self.rng.uniform(-2, 2), self.rng.uniform(-2, 2)),
                'life': self.rng.integers(15, 30),
                'max_life': 30,
                'color': pygame.Color(*self.COLOR_COIN, 255),
                'size': self.rng.integers(2, 5)
            })

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Platform Hopper")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()