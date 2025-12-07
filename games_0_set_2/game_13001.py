import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:05:44.311753
# Source Brief: brief_03001.md
# Brief Index: 3001
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A minimalist space shooter where players strategically switch between a precise
    laser and a rapid-fire cannon to destroy 20 enemy ships with limited ammunition.
    The game is designed as a Gymnasium environment with a focus on visual quality
    and satisfying game feel.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right for aiming)
    - actions[1]: Fire (0=released, 1=held)
    - actions[2]: Switch Weapon (0=released, 1=held)

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - An RGB image of the game.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A minimalist space shooter where you switch between a laser and a cannon to destroy enemy ships with limited ammo."
    )
    user_guide = (
        "Controls: ←→ to aim, space to fire, and shift to switch weapons. Destroy all enemies before running out of ammo."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30 # For physics calculations, not rendering speed

    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (220, 220, 255)
    COLOR_ENEMY = (255, 65, 54)
    COLOR_LASER = (46, 204, 64)
    COLOR_CANNON = (255, 220, 0)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_PARTICLE = (255, 255, 255)
    
    PLAYER_Y_POS = 380
    PLAYER_AIM_SPEED = 0.05  # Radians per step
    PLAYER_AIM_LIMIT = math.pi * 0.9 # From -limit/2 to +limit/2 around vertical

    MAX_AMMO = 15
    TOTAL_ENEMIES = 20
    MAX_EPISODE_STEPS = 1000

    WEAPON_LASER = 0
    WEAPON_CANNON = 1

    # Laser properties
    LASER_SPEED = 15
    LASER_COOLDOWN = 10 # steps
    
    # Cannon properties
    CANNON_SPEED = 25
    CANNON_COOLDOWN = 5 # steps
    CANNON_SPREAD = 0.1 # radians

    ENEMY_PATTERN_CHANGE_TIME = 30 * FPS # steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_weapon = pygame.font.Font(None, 22)
        
        # --- Game State Initialization (to avoid AttributeError) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.ammo = 0
        self.enemies_remaining = 0
        self.player_pos = (0, 0)
        self.player_aim_angle = 0.0
        self.current_weapon = self.WEAPON_LASER
        self.projectiles = []
        self.enemies = []
        self.particles = []
        self.cooldowns = {}
        self.pattern_timer = 0
        
        # self.reset() # reset is called by the environment runner
        # self.validate_implementation() # Not needed in production code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.ammo = self.MAX_AMMO
        self.enemies_remaining = self.TOTAL_ENEMIES
        
        self.player_pos = (self.SCREEN_WIDTH // 2, self.PLAYER_Y_POS)
        self.player_aim_angle = 0.0
        
        self.current_weapon = self.WEAPON_LASER
        
        self.projectiles = []
        self.enemies = self._spawn_enemies()
        self.particles = []
        
        self.cooldowns = {
            'fire': 0,
            'weapon_switch': 0
        }
        self.pattern_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0.0
        
        self._handle_input(action)
        collision_reward = self._update_game_state()
        step_reward += collision_reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_EPISODE_STEPS

        if terminated or truncated:
            self.game_over = True
            if self.enemies_remaining == 0 and not truncated:
                step_reward += 100.0 # Victory bonus
            elif self.ammo == 0 and not self.projectiles and not truncated:
                step_reward -= 100.0 # Failure penalty
        
        self.score += step_reward
        
        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Update Cooldowns ---
        for key in self.cooldowns:
            self.cooldowns[key] = max(0, self.cooldowns[key] - 1)

        # --- Aiming ---
        if movement == 3: # Left
            self.player_aim_angle -= self.PLAYER_AIM_SPEED
        if movement == 4: # Right
            self.player_aim_angle += self.PLAYER_AIM_SPEED
        
        # Clamp aim angle to be mostly upwards
        max_angle = self.PLAYER_AIM_LIMIT / 2
        self.player_aim_angle = np.clip(self.player_aim_angle, -max_angle, max_angle)

        # --- Firing ---
        if space_pressed and self.cooldowns['fire'] == 0 and self.ammo > 0:
            self.ammo -= 1
            # sfx: player_shoot.wav
            
            aim_vector_x = -math.sin(self.player_aim_angle)
            aim_vector_y = -math.cos(self.player_aim_angle)

            if self.current_weapon == self.WEAPON_LASER:
                self.cooldowns['fire'] = self.LASER_COOLDOWN
                projectile = {
                    'pos': list(self.player_pos),
                    'vel': [aim_vector_x * self.LASER_SPEED, aim_vector_y * self.LASER_SPEED],
                    'type': self.WEAPON_LASER,
                    'size': (4, 12)
                }
                self.projectiles.append(projectile)

            elif self.current_weapon == self.WEAPON_CANNON:
                self.cooldowns['fire'] = self.CANNON_COOLDOWN
                spread_angle = self.player_aim_angle + self.np_random.uniform(-self.CANNON_SPREAD, self.CANNON_SPREAD)
                vel_x = -math.sin(spread_angle) * self.CANNON_SPEED
                vel_y = -math.cos(spread_angle) * self.CANNON_SPEED
                
                projectile = {
                    'pos': list(self.player_pos),
                    'vel': [vel_x, vel_y],
                    'type': self.WEAPON_CANNON,
                    'size': 8 # radius
                }
                self.projectiles.append(projectile)

        # --- Weapon Switch ---
        if shift_pressed and self.cooldowns['weapon_switch'] == 0:
            self.current_weapon = 1 - self.current_weapon # Toggle 0 and 1
            self.cooldowns['weapon_switch'] = 15 # Prevent rapid switching
            # sfx: weapon_switch.wav

    def _update_game_state(self):
        # --- Update Projectiles ---
        for p in self.projectiles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
        
        self.projectiles = [p for p in self.projectiles if 0 < p['pos'][0] < self.SCREEN_WIDTH and p['pos'][1] > 0]
        
        # --- Update Enemies ---
        self.pattern_timer += 1
        if self.pattern_timer > self.ENEMY_PATTERN_CHANGE_TIME:
            self.pattern_timer = 0
            for enemy in self.enemies:
                enemy['amplitude'] = self.np_random.uniform(10, 50)
                enemy['frequency'] = self.np_random.uniform(0.02, 0.08)
                enemy['speed'] = self.np_random.uniform(0.5, 2.0) * self.np_random.choice([-1, 1])

        for enemy in self.enemies:
            enemy['pos'][0] += enemy['speed']
            enemy['pos'][1] = enemy['base_y'] + enemy['amplitude'] * math.sin(enemy['frequency'] * self.steps + enemy['phase'])
            if not (0 < enemy['pos'][0] < self.SCREEN_WIDTH - enemy['size']):
                enemy['speed'] *= -1
                enemy['pos'][0] = np.clip(enemy['pos'][0], 0, self.SCREEN_WIDTH - enemy['size'])
        
        # --- Update Particles ---
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

        # --- Handle Collisions ---
        return self._check_collisions()

    def _check_collisions(self):
        reward = 0.0
        projectiles_to_remove = set()
        enemies_to_remove = set()

        for i, p in enumerate(self.projectiles):
            proj_size_w = p['size'][0] if p['type'] == self.WEAPON_LASER else p['size']*2
            proj_size_h = p['size'][1] if p['type'] == self.WEAPON_LASER else p['size']*2
            proj_rect = pygame.Rect(p['pos'][0] - proj_size_w/2, p['pos'][1] - proj_size_h/2, proj_size_w, proj_size_h)
            
            for j, enemy in enumerate(self.enemies):
                if j in enemies_to_remove: continue

                enemy_rect = pygame.Rect(enemy['pos'][0], enemy['pos'][1], enemy['size'], enemy['size'])
                if proj_rect.colliderect(enemy_rect):
                    # sfx: explosion.wav
                    dist_to_center = math.hypot(p['pos'][0] - enemy_rect.centerx, p['pos'][1] - enemy_rect.centery)
                    
                    reward += 0.1 if dist_to_center < 5 else 0.05
                    reward += 10.0 # Destruction reward
                    
                    self._create_explosion(enemy_rect.center, 30)
                    self.enemies_remaining -= 1
                    
                    projectiles_to_remove.add(i)
                    enemies_to_remove.add(j)
                    break # Projectile can only hit one enemy
        
        if projectiles_to_remove or enemies_to_remove:
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
            self.enemies = [e for j, e in enumerate(self.enemies) if j not in enemies_to_remove]

        return reward

    def _check_termination(self):
        if self.enemies_remaining <= 0:
            return True
        if self.ammo <= 0 and not self.projectiles:
            return True
        # Truncation is handled separately in step()
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.ammo,
            "enemies_remaining": self.enemies_remaining,
        }

    def _spawn_enemies(self):
        enemies = []
        rows = 4
        cols = self.TOTAL_ENEMIES // rows
        for i in range(self.TOTAL_ENEMIES):
            row = i // cols
            col = i % cols
            
            x = (self.SCREEN_WIDTH / (cols + 1)) * (col + 1)
            y = 50 + row * 40

            enemy = {
                'pos': [x, y],
                'size': 16,
                'base_y': y,
                'amplitude': self.np_random.uniform(5, 20),
                'frequency': self.np_random.uniform(0.05, 0.1),
                'phase': self.np_random.uniform(0, 2 * math.pi),
                'speed': self.np_random.uniform(0.5, 1.5) * self.np_random.choice([-1, 1])
            }
            enemies.append(enemy)
        return enemies

    def _create_explosion(self, position, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            particle = {
                'pos': list(position),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': self.np_random.integers(10, 25)
            }
            self.particles.append(particle)

    def _render_game(self):
        # --- Render Particles ---
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 25))
            color = (*self.COLOR_PARTICLE, alpha)
            # Use a surface with SRCALPHA for proper blending
            temp_surf = pygame.Surface((2, 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (1, 1), 1)
            self.screen.blit(temp_surf, (int(p['pos'][0]) - 1, int(p['pos'][1]) - 1), special_flags=pygame.BLEND_RGBA_ADD)

        # --- Render Enemies ---
        for enemy in self.enemies:
            rect = pygame.Rect(int(enemy['pos'][0]), int(enemy['pos'][1]), enemy['size'], enemy['size'])
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect)
            pygame.draw.rect(self.screen, tuple(min(255, c+60) for c in self.COLOR_ENEMY), rect, 1)

        # --- Render Player ---
        player_points = [
            (self.player_pos[0] - 10, self.player_pos[1] + 8),
            (self.player_pos[0], self.player_pos[1] - 12),
            (self.player_pos[0] + 10, self.player_pos[1] + 8)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, player_points)
        pygame.draw.aalines(self.screen, (255,255,255), True, player_points)

        # --- Render Aiming Reticle ---
        aim_len = 30
        end_x = self.player_pos[0] - math.sin(self.player_aim_angle) * aim_len
        end_y = self.player_pos[1] - math.cos(self.player_aim_angle) * aim_len
        pygame.draw.aaline(self.screen, self.COLOR_PLAYER, self.player_pos, (end_x, end_y), 1)

        # --- Render Projectiles ---
        for p in self.projectiles:
            if p['type'] == self.WEAPON_LASER:
                angle_deg = math.degrees(math.atan2(-p['vel'][0], -p['vel'][1]))
                surf = pygame.Surface(p['size'], pygame.SRCALPHA)
                surf.fill(self.COLOR_LASER)
                rotated_surf = pygame.transform.rotate(surf, angle_deg)
                rect = rotated_surf.get_rect(center=(int(p['pos'][0]), int(p['pos'][1])))
                self.screen.blit(rotated_surf, rect)
            elif p['type'] == self.WEAPON_CANNON:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p['size'], self.COLOR_CANNON)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], p['size'], self.COLOR_CANNON)
                # Adding a glow effect for the cannon shot
                glow_surf = pygame.Surface((p['size']*4, p['size']*4), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*self.COLOR_CANNON, 50), (p['size']*2, p['size']*2), p['size']+2)
                self.screen.blit(glow_surf, (pos[0]-p['size']*2, pos[1]-p['size']*2), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_ui(self):
        # --- Ammo Count ---
        ammo_text = self.font_ui.render(f"AMMO: {self.ammo}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10, 10))
        
        # --- Enemy Count ---
        enemy_text = self.font_ui.render(f"ENEMIES: {self.enemies_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(enemy_text, (self.SCREEN_WIDTH - enemy_text.get_width() - 10, 10))

        # --- Current Weapon ---
        weapon_name = "LASER" if self.current_weapon == self.WEAPON_LASER else "CANNON"
        weapon_color = self.COLOR_LASER if self.current_weapon == self.WEAPON_LASER else self.COLOR_CANNON
        weapon_text = self.font_weapon.render(weapon_name, True, weapon_color)
        text_pos = (self.player_pos[0] - weapon_text.get_width() / 2, self.player_pos[1] + 15)
        self.screen.blit(weapon_text, text_pos)

        # --- Game Over ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            
            message = "VICTORY" if self.enemies_remaining == 0 else "DEFEAT"
            color = (100, 255, 100) if self.enemies_remaining == 0 else (255, 100, 100)
            
            font_large = pygame.font.Font(None, 72)
            game_over_text = font_large.render(message, True, color)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage (for testing by a human player) ---
if __name__ == '__main__':
    # Un-comment the line below to run with a display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0.0
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Minimalist Space Shooter")
    clock = pygame.time.Clock()
    
    # Game state for human play
    game_is_terminated = False
    
    while running:
        # --- Human Input ---
        movement = 0 # none
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0
                game_is_terminated = False

        # --- Gym Step ---
        if not game_is_terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                game_is_terminated = True
                print(f"Episode finished. Total Score: {info['score']:.2f}")

        # --- Render to screen ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
            
        clock.tick(GameEnv.FPS)

    env.close()