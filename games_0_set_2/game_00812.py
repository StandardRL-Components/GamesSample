
# Generated: 2025-08-27T14:52:07.400314
# Source Brief: brief_00812.md
# Brief Index: 812

        
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
        "Controls: Use arrow keys (↑↓←→) to move your ship. "
        "Press and hold [SPACE] to fire your laser cannon. "
        "Survive the alien onslaught!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro-arcade top-down space shooter. "
        "Destroy all 50 alien ships in the wave to win. You have 3 lives."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        self.TOTAL_ENEMIES = 50

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_OUTLINE = (128, 255, 200)
        self.COLOR_ENEMY_BASIC = (255, 50, 50)
        self.COLOR_ENEMY_SHIELDED = (200, 50, 255)
        self.COLOR_ENEMY_BOMBER = (255, 150, 50)
        self.COLOR_SHIELD = (100, 150, 255, 100) # RGBA
        self.COLOR_PROJECTILE_PLAYER = (255, 255, 255)
        self.COLOR_PROJECTILE_ENEMY = (255, 200, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_EXPLOSION = [(255, 255, 100), (255, 150, 50), (255, 50, 50)]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_wave = pygame.font.Font(None, 36)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_rect = None
        self.player_lives = 0
        self.player_fire_cooldown = 0
        self.player_projectiles = []
        self.enemies = []
        self.enemy_projectiles = []
        self.explosions = []
        self.stars = []
        self.enemies_destroyed = 0
        self.base_enemy_fire_rate = 0.0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_rect = pygame.Rect(self.WIDTH // 2 - 15, self.HEIGHT - 50, 30, 30)
        self.player_lives = 3
        self.player_fire_cooldown = 0

        self.player_projectiles = []
        self.enemy_projectiles = []
        self.explosions = []
        self.enemies_destroyed = 0
        self.base_enemy_fire_rate = 0.1 # shots per second

        self._spawn_stars()
        self._spawn_enemies()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action
        reward = -0.01  # Small penalty for existing

        # --- Update Cooldowns ---
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1

        # --- Handle Player Actions ---
        self._handle_player_movement(movement)
        if space_held and self.player_fire_cooldown == 0:
            self._fire_player_projectile()
            # sfx: player_shoot.wav

        # --- Update Game Objects ---
        reward += self._update_player_projectiles()
        self._update_enemies()
        reward += self._update_enemy_projectiles()
        self._update_explosions()
        self._update_stars()

        # --- Check Termination Conditions ---
        terminated = False
        if self.player_lives <= 0:
            reward -= 100
            terminated = True
        elif self.enemies_destroyed == self.TOTAL_ENEMIES:
            reward += 50
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _spawn_stars(self):
        self.stars = []
        for _ in range(150):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            speed = random.uniform(0.2, 1.0)
            size = int(speed * 2)
            self.stars.append([x, y, speed, size])

    def _spawn_enemies(self):
        self.enemies = []
        for i in range(self.TOTAL_ENEMIES):
            row = i // 10
            col = i % 10
            
            if i < 20: # First 20
                enemy_type = 'shielded' if i % 10 < 2 else 'basic'
            elif i < 40: # Next 20
                if i % 10 < 2: enemy_type = 'bomber'
                elif i % 10 < 6: enemy_type = 'shielded'
                else: enemy_type = 'basic'
            else: # Last 10
                if i % 10 < 3: enemy_type = 'bomber'
                elif i % 10 < 8: enemy_type = 'shielded'
                else: enemy_type = 'basic'

            hp_map = {'basic': 1, 'shielded': 2, 'bomber': 3}
            size_map = {'basic': (24, 24), 'shielded': (28, 28), 'bomber': (32, 32)}
            
            x = col * (self.WIDTH / 10) + (self.WIDTH / 20) - size_map[enemy_type][0] / 2
            y = -50 - row * 40
            
            self.enemies.append({
                'rect': pygame.Rect(x, y, *size_map[enemy_type]),
                'type': enemy_type,
                'hp': hp_map[enemy_type],
                'max_hp': hp_map[enemy_type],
                'path_t': i * 0.5,
                'fire_cooldown': random.randint(0, self.FPS * 2),
                'spawn_y': 50 + row * 40
            })

    def _handle_player_movement(self, movement):
        player_speed = 7
        if movement == 1: self.player_rect.y -= player_speed # Up
        if movement == 2: self.player_rect.y += player_speed # Down
        if movement == 3: self.player_rect.x -= player_speed # Left
        if movement == 4: self.player_rect.x += player_speed # Right
        
        self.player_rect.left = max(0, self.player_rect.left)
        self.player_rect.right = min(self.WIDTH, self.player_rect.right)
        self.player_rect.top = max(0, self.player_rect.top)
        self.player_rect.bottom = min(self.HEIGHT, self.player_rect.bottom)

    def _fire_player_projectile(self):
        self.player_fire_cooldown = 10 # 6 shots/sec
        proj_rect = pygame.Rect(self.player_rect.centerx - 2, self.player_rect.top - 10, 4, 10)
        self.player_projectiles.append(proj_rect)

    def _update_player_projectiles(self):
        reward = 0
        for proj in self.player_projectiles[:]:
            proj.y -= 15
            if proj.bottom < 0:
                self.player_projectiles.remove(proj)
                continue

            for enemy in self.enemies[:]:
                if proj.colliderect(enemy['rect']):
                    # sfx: enemy_hit.wav
                    reward += 0.1
                    enemy['hp'] -= 1
                    self._create_explosion(proj.midtop, 1, 5)
                    if proj in self.player_projectiles:
                        self.player_projectiles.remove(proj)

                    if enemy['hp'] <= 0:
                        reward_map = {'basic': 1, 'shielded': 2, 'bomber': 3}
                        reward += reward_map[enemy['type']]
                        self.score += reward_map[enemy['type']] * 10
                        self._create_explosion(enemy['rect'].center, 2, 20)
                        self.enemies.remove(enemy)
                        self.enemies_destroyed += 1
                        # sfx: enemy_explode.wav
                    break
        return reward

    def _update_enemies(self):
        current_fire_rate = self.base_enemy_fire_rate + self.enemies_destroyed * 0.01
        for enemy in self.enemies:
            # Movement
            if enemy['rect'].y < enemy['spawn_y']:
                enemy['rect'].y += 2
            else:
                enemy['path_t'] += 0.02
                amplitude = 100 if enemy['type'] != 'bomber' else 50
                enemy['rect'].centerx = (self.WIDTH / 2) + math.sin(enemy['path_t']) * amplitude
                enemy['rect'].y += 0.2

            # Firing
            if enemy['rect'].y > 0 and random.random() < current_fire_rate / self.FPS:
                self._fire_enemy_projectile(enemy)
                # sfx: enemy_shoot.wav

    def _fire_enemy_projectile(self, enemy):
        if enemy['type'] == 'bomber':
            proj_rect = pygame.Rect(enemy['rect'].centerx - 4, enemy['rect'].bottom, 8, 8)
            self.enemy_projectiles.append({'rect': proj_rect, 'speed': 4})
        else:
            proj_rect = pygame.Rect(enemy['rect'].centerx - 2, enemy['rect'].bottom, 4, 10)
            self.enemy_projectiles.append({'rect': proj_rect, 'speed': 6})

    def _update_enemy_projectiles(self):
        reward = 0
        for proj_data in self.enemy_projectiles[:]:
            proj_rect = proj_data['rect']
            proj_rect.y += proj_data['speed']
            if proj_rect.top > self.HEIGHT:
                self.enemy_projectiles.remove(proj_data)
                continue
            
            if not self.game_over and proj_rect.colliderect(self.player_rect):
                self.player_lives -= 1
                reward -= 1.0
                self._create_explosion(self.player_rect.center, 3, 30)
                self.enemy_projectiles.remove(proj_data)
                if self.player_lives > 0:
                    self.player_rect.center = (self.WIDTH // 2, self.HEIGHT - 50)
                # sfx: player_explode.wav
        return reward

    def _create_explosion(self, pos, scale, count):
        particles = []
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4) * scale
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(10, 20)
            size = random.uniform(1, 3) * scale
            color = random.choice(self.COLOR_EXPLOSION)
            particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'size': size, 'color': color})
        self.explosions.append({'particles': particles})

    def _update_explosions(self):
        for explosion in self.explosions[:]:
            for p in explosion['particles'][:]:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['lifetime'] -= 1
                p['size'] *= 0.95
                if p['lifetime'] <= 0:
                    explosion['particles'].remove(p)
            if not explosion['particles']:
                self.explosions.remove(explosion)

    def _update_stars(self):
        for star in self.stars:
            star[1] += star[2]
            if star[1] > self.HEIGHT:
                star[1] = 0
                star[0] = random.randint(0, self.WIDTH)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, speed, size in self.stars:
            color_val = int(speed * 100)
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(x), int(y)), int(size))
        
        # Enemy Projectiles
        for proj_data in self.enemy_projectiles:
            if proj_data['speed'] > 5: # Laser
                pygame.draw.rect(self.screen, self.COLOR_PROJECTILE_ENEMY, proj_data['rect'])
            else: # Bomb
                pygame.gfxdraw.filled_circle(self.screen, proj_data['rect'].centerx, proj_data['rect'].centery, proj_data['rect'].width // 2, self.COLOR_ENEMY_BOMBER)
        
        # Player Projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE_PLAYER, proj)

        # Enemies
        for enemy in self.enemies:
            color_map = {'basic': self.COLOR_ENEMY_BASIC, 'shielded': self.COLOR_ENEMY_SHIELDED, 'bomber': self.COLOR_ENEMY_BOMBER}
            pygame.draw.rect(self.screen, color_map[enemy['type']], enemy['rect'])
            if enemy['type'] == 'shielded' and enemy['hp'] == enemy['max_hp']:
                shield_surface = pygame.Surface(enemy['rect'].size, pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(shield_surface, enemy['rect'].width//2, enemy['rect'].height//2, enemy['rect'].width//2 + 2, self.COLOR_SHIELD)
                self.screen.blit(shield_surface, enemy['rect'].topleft)

        # Player Ship
        if self.player_lives > 0:
            p = self.player_rect
            points = [(p.centerx, p.top), (p.left, p.bottom), (p.right, p.bottom)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_OUTLINE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Explosions
        for explosion in self.explosions:
            for p in explosion['particles']:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.draw.circle(self.screen, p['color'], pos, int(p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 150, 10))
        for i in range(self.player_lives):
            ship_points = [
                (self.WIDTH - 80 + i * 25, 12),
                (self.WIDTH - 90 + i * 25, 28),
                (self.WIDTH - 70 + i * 25, 28)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, ship_points, self.COLOR_PLAYER)

        # Wave Info
        wave_text = self.font_wave.render(f"DESTROY {self.TOTAL_ENEMIES - self.enemies_destroyed} ALIENS", True, self.COLOR_TEXT)
        text_rect = wave_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 20))
        self.screen.blit(wave_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_lives": self.player_lives,
            "enemies_remaining": self.TOTAL_ENEMIES - self.enemies_destroyed,
        }

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Shooter Gym Environment")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    # --- Instructions ---
    print("\n" + "="*40)
    print(env.game_description)
    print(env.user_guide)
    print("="*40 + "\n")

    while running:
        # --- Action Mapping from Keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
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

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)

    env.close()