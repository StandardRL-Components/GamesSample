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
        "Controls: ←→ to move, SHIFT for long jump, SPACE for short jump. ↓ to fall faster."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Leap across procedurally generated platforms, collecting coins and dodging obstacles to reach the end of each stage in this side-scrolling arcade hopper."
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

        # Colors
        self.COLOR_BG_TOP = (20, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (150, 255, 150, 50)
        self.COLOR_PLATFORM = (150, 150, 150)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_PORTAL = (200, 100, 255)
        self.COLOR_TEXT = (255, 255, 255)

        # Game constants
        self.GRAVITY = 0.4
        self.PLAYER_JUMP_SHORT = -8
        self.PLAYER_JUMP_LONG = -11
        self.PLAYER_AIR_CONTROL = 0.5
        self.PLAYER_FAST_FALL = 1.0
        self.PLAYER_MAX_VX = 5
        self.FRICTION = 0.95
        self.INITIAL_LIVES = 3
        self.TIME_PER_STAGE = 60 * 30 # 60 seconds at 30fps
        self.MAX_STAGES = 3

        # Game state variables
        self.player_pos = None
        self.player_vel = None
        self.player_on_ground = None
        self.player_rect = None
        self.player_invincible_timer = None
        
        self.platforms = []
        self.coins = []
        self.obstacles = []
        self.particles = []
        
        self.camera_x = 0
        self.stage_end_x = 0
        
        self.score = 0
        self.lives = 0
        self.timer = 0
        self.stage = 0
        self.steps = 0
        self.game_over = False

        self._pixel_font = self._create_pixel_font()
        
        # self.reset() is called to initialize the state, which requires a seed.
        # We call it with a default seed here, but it will be properly seeded
        # by the environment wrapper later.
        self.reset(seed=0)
        
    def _setup_stage(self):
        self.platforms.clear()
        self.coins.clear()
        self.obstacles.clear()
        self.particles.clear()
        
        self.timer = self.TIME_PER_STAGE
        self.player_invincible_timer = 0
        
        # Create starting platform
        start_platform = pygame.Rect(50, self.HEIGHT - 50, 100, 20)
        self.platforms.append(start_platform)
        
        self.player_pos = [start_platform.centerx, start_platform.top - 20]
        self.player_vel = [0, 0]
        self.player_on_ground = True

        # FIX: Initialize player_rect here to avoid NoneType error on reset
        player_w, player_h = 20, 30
        self.player_rect = pygame.Rect(self.player_pos[0] - player_w/2, self.player_pos[1] - player_h, player_w, player_h)
        
        # Procedural generation
        current_x = start_platform.right
        max_jump_dist_x = abs(self.PLAYER_MAX_VX * (2 * self.PLAYER_JUMP_LONG / self.GRAVITY)) * 0.9
        
        stage_length = 3000 + self.stage * 1000
        while current_x < stage_length:
            gap = self.np_random.uniform(50, max_jump_dist_x * 0.6)
            width = self.np_random.integers(80, 200)
            
            last_y = self.platforms[-1].y
            y_diff = self.np_random.uniform(-80, 80)
            new_y = np.clip(last_y + y_diff, self.HEIGHT * 0.4, self.HEIGHT - 30)

            plat_x = current_x + gap
            platform = pygame.Rect(plat_x, new_y, width, 20)
            self.platforms.append(platform)

            # Add coins
            if self.np_random.random() < 0.6:
                num_coins = self.np_random.integers(1, 4)
                for i in range(num_coins):
                    coin_x = platform.x + (platform.width / (num_coins + 1)) * (i + 1)
                    coin_y = platform.y - 40 - self.np_random.uniform(0, 20)
                    self.coins.append(pygame.Rect(coin_x, coin_y, 10, 10))

            # Add obstacles
            if self.np_random.random() < 0.2 + self.stage * 0.1:
                obstacle_speed = 1.0 + (self.stage - 1) * 0.5
                bar = {
                    "rect": pygame.Rect(plat_x + width/2 - 10, new_y - 100, 20, 80),
                    "vel": [0, self.np_random.choice([-1,1]) * obstacle_speed],
                    "range": [new_y - 150, new_y - 20]
                }
                self.obstacles.append(bar)

            current_x = platform.right
            
        self.stage_end_x = current_x + 200
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.stage = 1
        self.steps = 0
        self.game_over = False
        self.camera_x = 0
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _handle_player_death(self):
        self.lives -= 1
        # sound: player_death.wav
        if self.lives <= 0:
            self.game_over = True
            return -10.0 # Final penalty for losing all lives
        
        for _ in range(30):
            self._create_particle(self.player_pos, self.COLOR_OBSTACLE, count=1, random_vel=4)
        
        start_platform = self.platforms[0]
        self.player_pos = [start_platform.centerx, start_platform.top - 20]
        self.player_vel = [0, 0]
        self.player_on_ground = True
        self.player_invincible_timer = 90 # 3 seconds of invincibility
        return -5.0 # Respawn penalty
        
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        reward = 0.1 # Survival reward

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Ground movement and jumping
        if self.player_on_ground:
            self.player_vel[0] *= 0.8
            jump_acted = False
            if shift_held:
                self.player_vel[1] = self.PLAYER_JUMP_LONG
                self.player_on_ground = False
                jump_acted = True
            elif space_held:
                self.player_vel[1] = self.PLAYER_JUMP_SHORT
                self.player_on_ground = False
                jump_acted = True
            
            if jump_acted:
                self._create_particle([self.player_rect.centerx, self.player_rect.bottom], 
                                      self.COLOR_PLATFORM, count=5, random_vel=2, lifetime=15)
        # Air control
        else:
            if movement == 3: # Left
                self.player_vel[0] -= self.PLAYER_AIR_CONTROL
            if movement == 4: # Right
                self.player_vel[0] += self.PLAYER_AIR_CONTROL
            if movement == 2: # Down
                self.player_vel[1] += self.PLAYER_FAST_FALL
        
        # Apply gravity
        if not self.player_on_ground:
            self.player_vel[1] += self.GRAVITY
        
        # Update position and velocity
        self.player_vel[0] = np.clip(self.player_vel[0], -self.PLAYER_MAX_VX, self.PLAYER_MAX_VX)
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        
        if not self.player_on_ground: self.player_vel[0] *= self.FRICTION
        
        # Player squash and stretch effect
        squash = 1 - min(max(self.player_vel[1] * 0.01, -0.4), 0.4)
        stretch = 1 + min(max(self.player_vel[1] * -0.01, -0.3), 0.5)
        player_w, player_h = 20 * squash, 30 * stretch
        self.player_rect = pygame.Rect(self.player_pos[0] - player_w/2, self.player_pos[1] - player_h, player_w, player_h)
        
        # Platform collision
        self.player_on_ground = False
        for plat in self.platforms:
            if self.player_rect.colliderect(plat) and self.player_vel[1] > 0:
                if (self.player_rect.bottom - self.player_vel[1]) <= plat.top:
                    self.player_pos[1] = plat.top
                    self.player_vel[1] = 0
                    self.player_on_ground = True
                    break
        
        # Coin collection
        collided_coins = []
        for i, coin in enumerate(self.coins):
            if self.player_rect.colliderect(coin):
                collided_coins.append(i)
                self.score += 1
                reward += 1.0
                self._create_particle(coin.center, self.COLOR_COIN, count=10, random_vel=3)

        for i in sorted(collided_coins, reverse=True): del self.coins[i]

        # Obstacle collision
        if self.player_invincible_timer > 0:
            self.player_invincible_timer -= 1
        else:
            for obs in self.obstacles:
                if self.player_rect.colliderect(obs["rect"]):
                    reward += self._handle_player_death()
                    break

        # Update obstacles
        for obs in self.obstacles:
            obs['rect'].y += obs['vel'][1]
            if obs['rect'].top < obs['range'][0] or obs['rect'].bottom > obs['range'][1]:
                obs['vel'][1] *= -1

        # Update particles
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1

        # Check for falling out of bounds
        if self.player_pos[1] > self.HEIGHT + 50:
            reward += self._handle_player_death()

        # Check for timeout
        if self.timer <= 0 and not self.game_over:
            reward += self._handle_player_death()

        # Check for stage completion
        if self.player_pos[0] > self.stage_end_x:
            self.stage += 1
            reward += 50.0
            if self.stage > self.MAX_STAGES:
                self.game_over = True
                reward += 100.0
            else:
                self._setup_stage()

        # Update camera
        target_camera_x = self.player_pos[0] - self.WIDTH / 2
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        
        terminated = self.game_over
        truncated = False # No explicit truncation condition in this logic
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _create_particle(self, pos, color, count, random_vel, lifetime=20):
        for _ in range(count):
            self.particles.append({
                "pos": list(pos),
                "vel": [self.np_random.uniform(-random_vel, random_vel), self.np_random.uniform(-random_vel, random_vel)],
                "lifetime": self.np_random.integers(lifetime // 2, lifetime),
                "color": color
            })
    
    def _render_game(self):
        cam_x_int = int(self.camera_x)

        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat.move(-cam_x_int, 0))

        for coin in self.coins:
            spin_factor = math.sin(self.steps * 0.2 + coin.x)
            w = int(coin.width * abs(spin_factor))
            h = coin.height
            c_rect = pygame.Rect(coin.x - w/2 + coin.width/2, coin.y, w, h)
            pygame.draw.ellipse(self.screen, self.COLOR_COIN, c_rect.move(-cam_x_int, 0))

        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs["rect"].move(-cam_x_int, 0))

        portal_x, portal_y = self.stage_end_x - cam_x_int, self.HEIGHT / 2
        for i in range(15, 0, -1):
            alpha = 150 - i * 10
            radius = 20 + i * 3
            color = (*self.COLOR_PORTAL, alpha)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(portal_x), int(portal_y), radius, color)
            except TypeError: # Sometimes color can have invalid alpha
                pass


        for p in self.particles:
            pos = (int(p['pos'][0] - cam_x_int), int(p['pos'][1]))
            size = max(1, int(p['lifetime'] * 0.2))
            pygame.draw.rect(self.screen, p['color'], (*pos, size, size))

        if self.lives > 0 and self.player_rect is not None:
            player_draw_rect = self.player_rect.copy()
            player_draw_rect.x -= cam_x_int
            
            if self.player_invincible_timer > 0 and self.steps % 6 < 3:
                pass
            else:
                glow_rect = player_draw_rect.inflate(10, 10)
                glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.ellipse(glow_surf, self.COLOR_PLAYER_GLOW, (0, 0, *glow_rect.size))
                self.screen.blit(glow_surf, glow_rect.topleft)
                
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_draw_rect, border_radius=5)
                
                eye_y = player_draw_rect.y + player_draw_rect.height * 0.3
                eye_l_x = player_draw_rect.centerx - player_draw_rect.width * 0.2
                eye_r_x = player_draw_rect.centerx + player_draw_rect.width * 0.2
                eye_size = max(2, int(player_draw_rect.height * 0.1))
                pygame.draw.circle(self.screen, (0,0,0), (eye_l_x, eye_y), eye_size)
                pygame.draw.circle(self.screen, (0,0,0), (eye_r_x, eye_y), eye_size)

    def _render_ui(self):
        self._render_pixel_text(f"SCORE {self.score}", 10, 10)
        self._render_pixel_text(f"TIME {max(0, self.timer // 30)}", self.WIDTH - 120, 10)
        self._render_pixel_text(f"LIVES {self.lives}", self.WIDTH // 2 - 50, 10)
        self._render_pixel_text(f"STAGE {self.stage}/{self.MAX_STAGES}", self.WIDTH // 2 - 60, 30)

        if self.game_over:
            msg = "YOU WIN!" if self.stage > self.MAX_STAGES else "GAME OVER"
            self._render_pixel_text(msg, self.WIDTH//2 - len(msg)*12, self.HEIGHT//2 - 20, scale=4)

    def _get_observation(self):
        # Draw gradient background
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp) for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives, "stage": self.stage}

    def _create_pixel_font(self):
        font_map = {'A':[1,1,1,1,0,1,1,1,1,1,0,1,1,0,1],'B':[1,1,0,1,0,1,1,1,0,1,0,1,1,1,0],'C':[0,1,1,1,0,0,1,0,0,1,0,0,0,1,1],'D':[1,1,0,1,0,1,1,0,1,1,0,1,1,1,0],'E':[1,1,1,1,0,0,1,1,0,1,0,0,1,1,1],'F':[1,1,1,1,0,0,1,1,0,1,0,0,1,0,0],'G':[0,1,1,1,0,0,1,0,1,1,0,1,0,1,1],'H':[1,0,1,1,0,1,1,1,1,1,0,1,1,0,1],'I':[1,1,1,0,1,0,0,1,0,0,1,0,1,1,1],'J':[0,0,1,0,0,1,0,0,1,1,0,1,0,1,0],'K':[1,0,1,1,0,1,1,1,0,1,0,1,1,0,1],'L':[1,0,0,1,0,0,1,0,0,1,0,0,1,1,1],'M':[1,0,1,1,1,1,1,1,1,1,0,1,1,0,1],'N':[1,0,1,1,1,1,1,1,1,1,1,1,1,0,1],'O':[0,1,0,1,0,1,1,0,1,1,0,1,0,1,0],'P':[1,1,0,1,0,1,1,1,0,1,0,0,1,0,0],'Q':[0,1,0,1,0,1,1,0,1,0,1,1,0,1,1],'R':[1,1,0,1,0,1,1,1,0,1,0,1,1,0,1],'S':[0,1,1,1,0,0,0,1,0,0,0,1,1,1,0],'T':[1,1,1,0,1,0,0,1,0,0,1,0,0,1,0],'U':[1,0,1,1,0,1,1,0,1,1,0,1,0,1,0],'V':[1,0,1,1,0,1,1,0,1,0,1,0,0,1,0],'W':[1,0,1,1,0,1,1,1,1,1,1,1,1,0,1],'X':[1,0,1,1,0,1,0,1,0,1,0,1,1,0,1],'Y':[1,0,1,1,0,1,0,1,0,0,1,0,0,1,0],'Z':[1,1,1,0,0,1,0,1,0,1,0,0,1,1,1],'0':[0,1,0,1,0,1,1,0,1,1,0,1,0,1,0],'1':[0,1,0,1,1,0,0,1,0,0,1,0,1,1,1],'2':[0,1,0,1,0,1,0,0,1,0,1,0,1,1,1],'3':[1,1,0,0,0,1,0,1,0,0,0,1,1,1,0],'4':[1,0,1,1,0,1,1,1,1,0,0,1,0,0,1],'5':[1,1,1,1,0,0,1,1,1,0,0,1,1,1,0],'6':[0,1,1,1,0,0,1,1,0,1,0,1,0,1,0],'7':[1,1,1,0,0,1,0,1,0,0,1,0,0,1,0],'8':[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],'9':[0,1,0,1,0,1,0,1,1,0,0,1,1,1,0],'!':[0,1,0,0,1,0,0,1,0,0,0,0,0,1,0],' ':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],'/':[0,0,1,0,0,1,0,1,0,1,0,0,1,0,0]}
        return font_map

    def _render_pixel_text(self, text, x, y, color=None, scale=2):
        color = color or self.COLOR_TEXT
        text = text.upper()
        char_width = 3 * scale
        spacing = 1 * scale
        
        for char in text:
            if char in self._pixel_font:
                pixels = self._pixel_font[char]
                for i, pixel in enumerate(pixels):
                    if pixel:
                        px = x + (i % 3) * scale
                        py = y + (i // 3) * scale
                        pygame.draw.rect(self.screen, color, (px, py, scale, scale))
            x += char_width + spacing

    def close(self):
        pygame.quit()

def validate_implementation(env_instance):
    print("Running implementation validation...")
    # Test action space
    assert isinstance(env_instance.action_space, MultiDiscrete)
    assert env_instance.action_space.nvec.tolist() == [5, 2, 2]
    
    # Test observation space  
    test_obs, _ = env_instance.reset()
    assert test_obs.shape == (env_instance.HEIGHT, env_instance.WIDTH, 3)
    assert test_obs.dtype == np.uint8
    
    # Test reset
    obs, info = env_instance.reset()
    assert obs.shape == (env_instance.HEIGHT, env_instance.WIDTH, 3)
    assert isinstance(info, dict)
    
    # Test step
    test_action = env_instance.action_space.sample()
    obs, reward, term, trunc, info = env_instance.step(test_action)
    assert obs.shape == (env_instance.HEIGHT, env_instance.WIDTH, 3)
    assert isinstance(reward, (int, float))
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)
    
    print("✓ Implementation validated successfully")

# Example usage for human playtesting
if __name__ == '__main__':
    # Switch the video driver back to a visible one for playtesting
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    # validate_implementation(env) # This is useful for headless testing
    
    obs, info = env.reset()
    
    pygame.display.set_caption("Arcade Hopper")
    screen_display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0.0
    
    while running:
        keys = pygame.key.get_pressed()
        
        # Action mapping: 0:none, 1:jump, 2:down, 3:left, 4:right
        movement = 0
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0.0
            
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()