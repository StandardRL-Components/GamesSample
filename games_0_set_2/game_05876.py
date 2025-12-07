
# Generated: 2025-08-28T06:21:20.612103
# Source Brief: brief_05876.md
# Brief Index: 5876

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Use ←→ to aim your jump. Use ↑↓ to adjust jump height. Press space to jump. Collect 50 coins to win!"
    )

    game_description = (
        "Hop between moving platforms, collecting coins to reach the target score before you fall."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)

        # Colors
        self.COLOR_BG_TOP = (10, 0, 30)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PLAYER = (57, 255, 20)
        self.COLOR_PLATFORM = (150, 150, 170)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_AIM = (255, 255, 255, 100)

        # Game constants
        self.GRAVITY = 0.4
        self.PLAYER_SIZE = 20
        self.MAX_STEPS = 2000
        self.WIN_SCORE = 50
        self.JUMP_V_BASE = 10
        self.JUMP_V_MOD = 2.5
        self.JUMP_H = 7
        self.MAX_PLATFORMS = 15
        
        # State variables are initialized in reset()
        self.player = {}
        self.platforms = []
        self.coins = []
        self.particles = []
        self.camera_y = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.speed_multiplier = 1.0
        self.highest_platform_y = 0
        self.last_jump_info = {}
        self.jump_aim = {'vx': 0, 'vy': self.JUMP_V_BASE}

        self.reset()
        
        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.speed_multiplier = 1.0
        self.last_space_held = False
        self.particles = []
        
        # Player state
        self.player = {
            'x': self.WIDTH / 2, 'y': 50,
            'vx': 0, 'vy': 0,
            'on_platform': True,
            'last_platform_y': 0
        }
        self.jump_aim = {'vx': 0, 'vy': self.JUMP_V_BASE}
        
        # Camera and world generation
        self.camera_y = 0
        self.platforms = []
        self.coins = []
        self._generate_initial_platforms()
        self.highest_platform_y = max(p['rect'].top for p in self.platforms) if self.platforms else 0

        return self._get_observation(), self._get_info()

    def _generate_initial_platforms(self):
        # Create a stable starting area
        start_platform = pygame.Rect(self.WIDTH / 2 - 75, self.HEIGHT - 50, 150, 20)
        self.platforms.append({'rect': start_platform, 'speed': 0, 'has_coin': False})
        
        # Procedurally generate the first few platforms
        last_y = start_platform.y
        last_x = start_platform.centerx
        for i in range(self.MAX_PLATFORMS - 1):
            w = self.np_random.integers(80, 150)
            h = 20
            # Position new platforms relative to the last
            px = last_x + self.np_random.uniform(-180, 180)
            py = last_y - self.np_random.uniform(60, 120)
            
            # Clamp to world bounds for initial generation
            px = np.clip(px, w/2, self.WIDTH - w/2)

            speed = self.np_random.choice([-1.5, -1, 1, 1.5])
            rect = pygame.Rect(px - w / 2, py, w, h)
            has_coin = self.np_random.random() < 0.4
            
            self.platforms.append({'rect': rect, 'speed': speed, 'has_coin': has_coin})
            if has_coin:
                self.coins.append(pygame.Rect(rect.centerx - 5, rect.top - 15, 10, 10))
            
            last_y = py
            last_x = px

    def _manage_world_elements(self):
        # Update and remove off-screen platforms
        visible_platforms = []
        for p in self.platforms:
            p['rect'].x += p['speed'] * self.speed_multiplier
            # Wrap platforms around horizontally
            if p['rect'].right < self.player['x'] - self.WIDTH * 1.5:
                p['rect'].left = self.player['x'] + self.WIDTH * 1.5
            elif p['rect'].left > self.player['x'] + self.WIDTH * 1.5:
                p['rect'].right = self.player['x'] - self.WIDTH * 1.5
            
            # Keep if vertically visible
            if p['rect'].top < self.camera_y + self.HEIGHT + 100:
                visible_platforms.append(p)
        self.platforms = visible_platforms
        
        # Remove collected coins
        self.coins = [c for c in self.coins if c.top < self.camera_y + self.HEIGHT + 100]

        # Spawn new platforms if needed
        self.highest_platform_y = min([p['rect'].top for p in self.platforms] + [self.player['y']])
        
        while len(self.platforms) < self.MAX_PLATFORMS:
            w = self.np_random.integers(80, 200)
            h = 20
            px = self.player['x'] + self.np_random.uniform(-self.WIDTH, self.WIDTH)
            py = self.highest_platform_y - self.np_random.uniform(80, 160)
            
            speed_options = [-2, -1.5, -1, 1, 1.5, 2]
            speed = self.np_random.choice(speed_options)
            rect = pygame.Rect(px - w / 2, py, w, h)
            has_coin = self.np_random.random() < 0.5

            self.platforms.append({'rect': rect, 'speed': speed, 'has_coin': has_coin})
            if has_coin:
                self.coins.append(pygame.Rect(rect.centerx - 5, rect.top - 15, 10, 10))
            
            self.highest_platform_y = min(self.highest_platform_y, py)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.01  # Small reward for surviving
        
        # 1. Unpack action and handle controls
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        jump_triggered = space_held and not self.last_space_held

        if self.player['on_platform']:
            # Adjust jump aim
            self.jump_aim['vy'] = self.JUMP_V_BASE
            if movement == 1: self.jump_aim['vy'] += self.JUMP_V_MOD
            if movement == 2: self.jump_aim['vy'] = max(2, self.jump_aim['vy'] - self.JUMP_V_MOD)
            
            self.jump_aim['vx'] = 0
            if movement == 3: self.jump_aim['vx'] = -self.JUMP_H
            if movement == 4: self.jump_aim['vx'] = self.JUMP_H
            
            if jump_triggered:
                # Store pre-jump info for reward calculation
                self.last_jump_info = {
                    'start_y': self.player['y'],
                    'platform_speed': self.player['vx']
                }
                
                self.player['vy'] = -self.jump_aim['vy']
                self.player['vx'] += self.jump_aim['vx']
                self.player['on_platform'] = False
                # Sound: Jump
                self._create_particles(self.player['x'], self.player['y'], 10, self.COLOR_PLAYER)

        self.last_space_held = space_held

        # 2. Apply physics
        if not self.player['on_platform']:
            self.player['vy'] += self.GRAVITY
            self.player['y'] += self.player['vy']
            self.player['x'] += self.player['vx']
        
        # 3. Handle collisions
        player_rect = pygame.Rect(self.player['x'] - self.PLAYER_SIZE/2, self.player['y'] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Platform collision (landing)
        if self.player['vy'] > 0:
            for p in self.platforms:
                if p['rect'].colliderect(player_rect) and self.player['y'] < p['rect'].centery:
                    self.player['y'] = p['rect'].top - self.PLAYER_SIZE / 2
                    self.player['vy'] = 0
                    self.player['vx'] = p['speed'] * self.speed_multiplier
                    self.player['on_platform'] = True
                    
                    # Reward for landing
                    # Penalty for landing on platform moving against jump direction
                    jump_dir = np.sign(self.jump_aim['vx'])
                    platform_dir = np.sign(p['speed'])
                    if jump_dir != 0 and platform_dir != 0 and jump_dir != platform_dir:
                        reward -= 0.2
                    
                    # Reward for large vertical jumps
                    if self.last_jump_info:
                        height_diff = self.last_jump_info['start_y'] - self.player['y']
                        if abs(height_diff) > 100:
                            reward += 2.0

                    # Sound: Land
                    self._create_particles(self.player['x'], self.player['y'] + self.PLAYER_SIZE/2, 5, self.COLOR_PLATFORM)
                    break
        
        # Coin collision
        for coin in self.coins[:]:
            if coin.colliderect(player_rect):
                self.coins.remove(coin)
                self.score += 1
                reward += 1.0
                # Sound: Coin collect
                self._create_particles(coin.centerx, coin.centery, 15, self.COLOR_COIN)
                
                # Update difficulty
                if self.score > 0 and self.score % 10 == 0:
                    self.speed_multiplier += 0.05

        # 4. Update world
        self._manage_world_elements()
        self._update_particles()
        
        # 5. Check for termination
        self.steps += 1
        terminated = self._check_termination()
        if terminated:
            if self.player['y'] > self.camera_y + self.HEIGHT + 50:
                reward = -10.0 # Fell off screen
            elif self.score >= self.WIN_SCORE:
                reward = 100.0 # Won
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        # Fell off screen (y increases downwards in camera space)
        if self.player['y'] > self.camera_y + self.HEIGHT + 50:
            self.game_over = True
            return True
        # Reached win score
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            return True
        # Exceeded max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _world_to_screen(self, x, y):
        # Player is always in the horizontal center of the screen
        screen_x = x - self.player['x'] + self.WIDTH / 2
        screen_y = y - self.camera_y
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        # Update camera smoothly
        target_cam_y = self.player['y'] - self.HEIGHT * 0.6
        self.camera_y += (target_cam_y - self.camera_y) * 0.1
        
        # Draw background gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            sx, sy = self._world_to_screen(p['x'], p['y'])
            pygame.draw.circle(self.screen, p['color'], (sx, sy), int(p['size']))

        # Draw platforms
        for p in self.platforms:
            sx, sy = self._world_to_screen(p['rect'].x, p['rect'].y)
            s_rect = pygame.Rect(sx, sy, p['rect'].width, p['rect'].height)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, s_rect, border_radius=3)

        # Draw coins
        for coin in self.coins:
            sx, sy = self._world_to_screen(coin.x, coin.y)
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, sx + coin.width//2, sy + coin.height//2, 10, self.COLOR_COIN + (50,))
            pygame.gfxdraw.filled_circle(self.screen, sx + coin.width//2, sy + coin.height//2, 7, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, sx + coin.width//2, sy + coin.height//2, 7, self.COLOR_COIN)

        # Draw player
        px, py = self._world_to_screen(self.player['x'], self.player['y'])
        player_rect = pygame.Rect(px - self.PLAYER_SIZE/2, py - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        # Glow effect
        glow_rect = player_rect.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_PLAYER + (50,), s.get_rect(), border_radius=8)
        self.screen.blit(s, glow_rect.topleft)
        # Main shape
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        
        # Draw aiming indicator if on platform
        if self.player['on_platform']:
            start_pos = (px, py)
            end_pos = (px + self.jump_aim['vx'] * 5, py - self.jump_aim['vy'])
            pygame.draw.line(self.screen, self.COLOR_AIM, start_pos, end_pos, 2)
            pygame.draw.circle(self.screen, self.COLOR_AIM, end_pos, 4)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
    
    def _create_particles(self, x, y, count, color):
        for _ in range(count):
            self.particles.append({
                'x': x, 'y': y,
                'vx': self.np_random.uniform(-2, 2),
                'vy': self.np_random.uniform(-3, 1),
                'size': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(15, 30),
                'color': color
            })
            
    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += self.GRAVITY * 0.5
            p['size'] -= 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]

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

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Change to 'windows' or 'Quartz' as needed

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Hopper")
    
    terminated = False
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op

    print("\n" + "="*30)
    print("      SPACE HOPPER      ")
    print("="*30)
    print(env.game_description)
    print("\n" + env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        mov = 0 # No movement
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Match FPS for smooth human play

    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    env.close()