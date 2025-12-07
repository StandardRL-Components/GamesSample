import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Avoid red hazards and collect yellow coins as you climb."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Leap across procedurally generated platforms, collecting coins and dodging falling hazards to reach the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.WIN_PLATFORM_Y = -5000

        # Colors
        self.COLOR_BG_TOP = (40, 50, 80)
        self.COLOR_BG_BOTTOM = (10, 20, 40)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255, 100)
        self.COLOR_PLATFORM = (180, 180, 190)
        self.COLOR_PLATFORM_TOP = (220, 220, 230)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_COIN_GLOW = (255, 223, 0, 100)
        self.COLOR_HAZARD = (255, 50, 50)
        self.COLOR_HAZARD_GLOW = (255, 100, 100, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_WIN_PLATFORM = (0, 255, 128)

        # Physics and Game Parameters
        self.GRAVITY = 0.8
        self.PLAYER_JUMP_STRENGTH = -14
        self.PLAYER_HORIZONTAL_ACCEL = 1.5
        self.PLAYER_FRICTION = 0.85
        self.MAX_PLAYER_VX = 8
        self.CAMERA_LERP_FACTOR = 0.08
        self.INITIAL_HAZARD_SPAWN_RATE = 0.02

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Internal state - these will be reset in self.reset()
        self.player = None
        self.platforms = []
        self.coins = []
        self.hazards = []
        self.particles = []
        self.camera_y = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.hazard_spawn_rate = self.INITIAL_HAZARD_SPAWN_RATE
        self.highest_y = self.HEIGHT
        
        self.np_random = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_y = 0
        
        # Player State
        self.player = {
            "rect": pygame.Rect(self.WIDTH // 2 - 12, self.HEIGHT - 80, 24, 24),
            "vel": pygame.Vector2(0, 0),
            "on_ground": False
        }
        
        # Clear entity lists
        self.platforms.clear()
        self.coins.clear()
        self.hazards.clear()
        self.particles.clear()
        
        # World Generation
        self._generate_initial_world()
        self.highest_y = self.player['rect'].y

        # Difficulty
        self.hazard_spawn_rate = self.INITIAL_HAZARD_SPAWN_RATE

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = False
        truncated = False

        # 1. Handle Input & Player Movement
        self._update_player(action)

        # 2. Update Entities
        self._update_hazards()
        self._update_particles()
        
        # 3. Handle Collisions
        coin_reward = self._handle_collisions()
        reward += coin_reward
        
        # 4. Manage World & Camera
        self._manage_world_scroll()
        
        # 5. Calculate Rewards
        y_change = self.highest_y - self.player['rect'].y
        if y_change > 0:
            reward += 0.1
            self.highest_y = self.player['rect'].y
        elif self.player['vel'].y > 1: # Penalize falling
             reward -= 0.01

        # 6. Check Termination Conditions
        if self.player['rect'].top > self.camera_y + self.HEIGHT:
            terminated = True
            reward -= 5
        
        if any(self.player['rect'].colliderect(h['rect']) for h in self.hazards):
            terminated = True
            reward = -5
            self._create_particles(self.player['rect'].center, self.COLOR_PLAYER, 30)
        
        for p in self.platforms:
            if p.get('is_win_platform', False) and self.player['rect'].colliderect(p['rect']) and self.player['vel'].y >= 0:
                terminated = True
                reward = 100
                self.score += 100 # Bonus score for winning
                self._create_particles(self.player['rect'].center, self.COLOR_WIN_PLATFORM, 50)
                break

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            truncated = True

        self.game_over = terminated or truncated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _update_player(self, action):
        movement = action[0]
        
        if movement == 3: # Left
            self.player['vel'].x -= self.PLAYER_HORIZONTAL_ACCEL
        elif movement == 4: # Right
            self.player['vel'].x += self.PLAYER_HORIZONTAL_ACCEL
        
        self.player['vel'].x *= self.PLAYER_FRICTION
        self.player['vel'].x = max(-self.MAX_PLAYER_VX, min(self.MAX_PLAYER_VX, self.player['vel'].x))
        if abs(self.player['vel'].x) < 0.1: self.player['vel'].x = 0

        self.player['rect'].x += self.player['vel'].x
        
        if self.player['rect'].left < 0:
            self.player['rect'].left = 0
            self.player['vel'].x = 0
        if self.player['rect'].right > self.WIDTH:
            self.player['rect'].right = self.WIDTH
            self.player['vel'].x = 0
            
        # Jump based on last frame's on_ground status
        if movement == 1 and self.player['on_ground']:
            self.player['vel'].y = self.PLAYER_JUMP_STRENGTH
            self._create_particles(self.player['rect'].midbottom, self.COLOR_PLATFORM_TOP, 10, -2)

        # Apply gravity
        self.player['vel'].y += self.GRAVITY
        
        # Update vertical position
        self.player['rect'].y += self.player['vel'].y

    def _handle_collisions(self):
        player_rect = self.player['rect']
        player_vel = self.player['vel']
        
        # Reset on_ground status. It will be set to True if a landing occurs.
        self.player['on_ground'] = False
        
        for p in self.platforms:
            plat_rect = p['rect']
            # If player is falling and collides with a platform, land on it.
            # The original complex check was prone to float/int errors.
            if player_rect.colliderect(plat_rect) and player_vel.y > 0:
                player_rect.bottom = plat_rect.top
                player_vel.y = 0
                self.player['on_ground'] = True

        collected_reward = 0
        for coin in self.coins[:]:
            if player_rect.colliderect(coin['rect']):
                self.coins.remove(coin)
                self.score += 1
                collected_reward += 1
                self._create_particles(coin['rect'].center, self.COLOR_COIN, 15)
        return collected_reward

    def _update_hazards(self):
        for h in self.hazards:
            h['rect'].y += h['speed']
        
        self.hazards = [h for h in self.hazards if h['rect'].top < self.camera_y + self.HEIGHT]
        
        if self.np_random.random() < self.hazard_spawn_rate and self.player['rect'].y < self.WIN_PLATFORM_Y + self.HEIGHT:
            x_pos = self.np_random.integers(0, self.WIDTH - 20)
            y_pos = self.camera_y - 50
            speed = self.np_random.integers(3, 7)
            self.hazards.append({'rect': pygame.Rect(x_pos, y_pos, 20, 20), 'speed': speed})

        if self.steps > 0 and self.steps % 200 == 0:
            self.hazard_spawn_rate = min(0.1, self.hazard_spawn_rate + 0.01)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _manage_world_scroll(self):
        target_camera_y = self.player['rect'].y - self.HEIGHT * 0.4
        self.camera_y += (target_camera_y - self.camera_y) * self.CAMERA_LERP_FACTOR
        
        if not self.platforms: return
        last_platform_y = min(p['rect'].y for p in self.platforms)
        
        while last_platform_y > self.camera_y - 50:
            if last_platform_y < self.WIN_PLATFORM_Y + 100: break # Stop generating near win platform

            prev_platform = min(self.platforms, key=lambda p: abs(p['rect'].y - last_platform_y))
            new_y = last_platform_y - self.np_random.integers(60, 100)
            new_width = self.np_random.integers(80, 150)
            
            max_h_dist = int(abs(self.PLAYER_JUMP_STRENGTH) * self.MAX_PLAYER_VX * 0.8)
            new_x_min = max(0, prev_platform['rect'].centerx - max_h_dist)
            new_x_max = min(self.WIDTH - new_width, prev_platform['rect'].centerx + max_h_dist)
            if new_x_min >= new_x_max:
                new_x = max(0, min(self.WIDTH - new_width, prev_platform['rect'].centerx - new_width//2))
            else:
                new_x = self.np_random.integers(new_x_min, new_x_max)

            self.platforms.append({'rect': pygame.Rect(new_x, new_y, new_width, 20)})
            
            if self.np_random.random() < 0.5:
                coin_x = new_x + new_width / 2 - 10
                coin_y = new_y - 30
                self.coins.append({'rect': pygame.Rect(coin_x, coin_y, 20, 20), 'angle': 0})
            
            last_platform_y = new_y

        self.platforms = [p for p in self.platforms if p['rect'].top < self.camera_y + self.HEIGHT + 50]
        self.coins = [c for c in self.coins if c['rect'].top < self.camera_y + self.HEIGHT + 50]

    def _generate_initial_world(self):
        start_plat_y = self.HEIGHT - 60
        self.platforms.append({'rect': pygame.Rect(self.WIDTH // 2 - 50, start_plat_y, 100, 20)})
        
        current_y = start_plat_y
        for _ in range(15):
            current_y -= self.np_random.integers(60, 100)
            x = self.np_random.integers(0, self.WIDTH - 100)
            width = self.np_random.integers(80, 150)
            self.platforms.append({'rect': pygame.Rect(x, current_y, width, 20)})
        
        self.platforms.append({
            'rect': pygame.Rect(self.WIDTH // 2 - 75, self.WIN_PLATFORM_Y, 150, 30),
            'is_win_platform': True
        })

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp) for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        cam_y = int(self.camera_y)

        for p in self.platforms:
            rect = p['rect'].move(0, -cam_y)
            color = self.COLOR_WIN_PLATFORM if p.get('is_win_platform') else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, (rect.x, rect.y, rect.width, 5), border_radius=3)

        for coin in self.coins:
            rect = coin['rect'].move(0, -cam_y)
            glow_surface = pygame.Surface((rect.width*2, rect.height*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, self.COLOR_COIN_GLOW, (rect.width, rect.height), rect.width)
            self.screen.blit(glow_surface, (rect.x - rect.width/2, rect.y - rect.height/2))
            pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, rect.width // 2, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, rect.width // 2, self.COLOR_COIN)
            
        for h in self.hazards:
            rect = h['rect'].move(0, -cam_y)
            points = [(rect.midtop), (rect.bottomleft), (rect.bottomright)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HAZARD)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HAZARD)

        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y - cam_y))
            radius = int(p['life'] / p['max_life'] * p['size'])
            if radius > 0:
                pygame.draw.circle(self.screen, p['color'], pos, radius)

        player_rect_cam = self.player['rect'].move(0, -cam_y)
        glow_surface = pygame.Surface((player_rect_cam.width*2, player_rect_cam.height*2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, self.COLOR_PLAYER_GLOW, glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, (player_rect_cam.x - player_rect_cam.width/2, player_rect_cam.y - player_rect_cam.height/2))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect_cam, border_radius=5)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_small.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _create_particles(self, pos, color, count, y_vel_mod=1):
        for _ in range(count):
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-4, 1) * y_vel_mod),
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(3, 7)
            })

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # The following code is for human interaction and visualization.
    # It is not part of the Gymnasium environment itself.
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Platformer Game")
    
    running = True
    total_reward = 0
    
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_UP]:
            action[0] = 1

        if keys[pygame.K_r]: # Reset game
             obs, info = env.reset()
             total_reward = 0
             print("--- Game Reset ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000)

    env.close()