
# Generated: 2025-08-28T00:48:43.264211
# Source Brief: brief_03908.md
# Brief Index: 3908

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game entities
class Player:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vx, self.vy = 0, 0
        self.width, self.height = 24, 40
        self.on_ground = False
        self.health = 100
        self.max_health = 100
        self.ammo = 30
        self.max_ammo = 30
        self.facing_right = True
        self.shoot_cooldown = 0
        self.invulnerable_timer = 0

class Enemy:
    def __init__(self, x, y, patrol_range):
        self.x, self.y = x, y
        self.vx, self.vy = 1, 0
        self.width, self.height = 28, 32
        self.on_ground = False
        self.health = 50
        self.max_health = 50
        self.start_x = x
        self.patrol_range = patrol_range
        self.shoot_cooldown = random.randint(60, 120)

class Projectile:
    def __init__(self, x, y, vx, vy, is_player_bullet):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.is_player_bullet = is_player_bullet
        self.life = 180 # 3 seconds

class Particle:
    def __init__(self, x, y, vx, vy, life, color, size):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.life = life
        self.max_life = life
        self.color = color
        self.size = size

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Hold Shift to aim up. Press Space to fire your weapon."
    )

    game_description = (
        "Pilot a combat robot through a futuristic cityscape, blasting enemy drones to achieve victory before time runs out."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH = self.WIDTH * 3
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 64)

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GROUND = (50, 55, 70)
        self.COLOR_PLAYER = (60, 140, 255)
        self.COLOR_PLAYER_DMG = (255, 100, 100)
        self.COLOR_ENEMY = (255, 80, 80)
        self.COLOR_PLAYER_BULLET = (255, 255, 100)
        self.COLOR_ENEMY_BULLET = (255, 100, 200)
        self.COLOR_HEALTH_GREEN = (100, 220, 100)
        self.COLOR_HEALTH_RED = (220, 100, 100)
        self.COLOR_UI_TEXT = (230, 230, 230)
        
        # Game constants
        self.MAX_STEPS = 3600 # 60 seconds at 60 FPS
        self.GRAVITY = 0.5
        self.GROUND_Y = self.HEIGHT - 50
        self.PLAYER_SPEED = 4
        self.PLAYER_JUMP_STRENGTH = -10

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.player = None
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.background_buildings = []
        self.camera_x = 0
        self.camera_shake = 0
        self.was_space_held = False
        self.rng = None

        self.reset()
        self.validate_implementation(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.camera_x = 0
        self.camera_shake = 0
        self.was_space_held = False

        player_start_x = self.WORLD_WIDTH / 2
        self.player = Player(player_start_x, self.GROUND_Y - 40)

        self.enemies = []
        enemy_positions = self.rng.choice(np.arange(300, self.WORLD_WIDTH - 300, 200), 5, replace=False)
        for x_pos in enemy_positions:
            self.enemies.append(Enemy(x_pos, self.GROUND_Y - 32, self.rng.integers(100, 250)))

        self.projectiles = []
        self.particles = []
        
        self._generate_background()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(60)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Time penalty

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        if movement == 3: # Left
            self.player.vx = -self.PLAYER_SPEED
            self.player.facing_right = False
        elif movement == 4: # Right
            self.player.vx = self.PLAYER_SPEED
            self.player.facing_right = True
        else:
            self.player.vx = 0

        # Jump
        if movement == 1 and self.player.on_ground:
            self.player.vy = self.PLAYER_JUMP_STRENGTH
            self.player.on_ground = False
            # Sound: Jump

        # Shoot
        if space_held and not self.was_space_held and self.player.ammo > 0 and self.player.shoot_cooldown <= 0:
            self.player.ammo -= 1
            self.player.shoot_cooldown = 10 # 6 shots per second
            bullet_vx = 10 if self.player.facing_right else -10
            bullet_vy = -5 if shift_held else 0
            self.projectiles.append(Projectile(self.player.x, self.player.y - self.player.height/2, bullet_vx, bullet_vy, True))
            # Muzzle flash
            flash_x = self.player.x + (20 if self.player.facing_right else -20)
            self.particles.append(Particle(flash_x, self.player.y - 20, 0, 0, 4, self.COLOR_PLAYER_BULLET, 10))
            # Sound: Shoot

        self.was_space_held = space_held
        
        # --- Update Game State ---
        
        # Player
        self.player.shoot_cooldown = max(0, self.player.shoot_cooldown - 1)
        self.player.invulnerable_timer = max(0, self.player.invulnerable_timer - 1)
        self.player.vy += self.GRAVITY
        self.player.x += self.player.vx
        self.player.y += self.player.vy
        self.player.x = np.clip(self.player.x, 0, self.WORLD_WIDTH - self.player.width)
        
        if self.player.y >= self.GROUND_Y - self.player.height:
            self.player.y = self.GROUND_Y - self.player.height
            self.player.vy = 0
            self.player.on_ground = True
        else:
            self.player.on_ground = False
            
        # Enemies
        for enemy in self.enemies:
            enemy.x += enemy.vx
            if abs(enemy.x - enemy.start_x) > enemy.patrol_range:
                enemy.vx *= -1

            enemy.shoot_cooldown -= 1
            if enemy.shoot_cooldown <= 0:
                enemy.shoot_cooldown = self.rng.integers(120, 240)
                dx = self.player.x - enemy.x
                dy = (self.player.y - self.player.height/2) - (enemy.y - enemy.height/2)
                dist = max(1, math.sqrt(dx**2 + dy**2))
                bullet_vx = (dx / dist) * 6
                bullet_vy = (dy / dist) * 6
                self.projectiles.append(Projectile(enemy.x, enemy.y - enemy.height/2, bullet_vx, bullet_vy, False))
                # Sound: Enemy Shoot
        
        # Projectiles
        projectiles_to_keep = []
        for p in self.projectiles:
            p.x += p.vx
            p.y += p.vy
            p.life -= 1
            
            hit = False
            if p.is_player_bullet:
                for enemy in self.enemies:
                    if self._check_collision(p, enemy):
                        enemy.health -= 25
                        reward += 0.1
                        self._create_explosion(p.x, p.y, self.COLOR_ENEMY, 5, 8)
                        hit = True
                        break
            else: # Enemy bullet
                if self.player.invulnerable_timer <= 0 and self._check_collision(p, self.player):
                    self.player.health -= 10
                    self.player.invulnerable_timer = 60 # 1s invulnerability
                    self.camera_shake = 10
                    reward -= 1
                    self._create_explosion(p.x, p.y, self.COLOR_PLAYER, 5, 8)
                    hit = True
            
            if not hit and p.life > 0 and 0 < p.x < self.WORLD_WIDTH and 0 < p.y < self.HEIGHT:
                projectiles_to_keep.append(p)
        self.projectiles = projectiles_to_keep

        # Check for defeated enemies
        enemies_alive = []
        for enemy in self.enemies:
            if enemy.health > 0:
                enemies_alive.append(enemy)
            else:
                reward += 10
                self._create_explosion(enemy.x + enemy.width/2, enemy.y + enemy.height/2, self.COLOR_ENEMY, 15, 25)
                self.camera_shake = 15
                # Sound: Explosion
        
        num_defeated = len(self.enemies) - len(enemies_alive)
        if num_defeated > 0:
            self.score += 100 * num_defeated
        self.enemies = enemies_alive
        
        # Particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.life -= 1
            p.x += p.vx
            p.y += p.vy

        # Update score
        self.score += reward

        # Update camera
        target_cam_x = self.player.x - self.WIDTH / 2
        self.camera_x += (target_cam_x - self.camera_x) * 0.1
        self.camera_x = np.clip(self.camera_x, 0, self.WORLD_WIDTH - self.WIDTH)
        
        if self.camera_shake > 0:
            self.camera_shake -= 1

        # --- Termination Check ---
        self.steps += 1
        terminated = False
        if self.player.health <= 0:
            reward -= 20
            self.score -= 20
            terminated = True
            self.game_over = True
            # Sound: Player Defeated
        elif not self.enemies:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
            self.game_won = True
            # Sound: Victory
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_background(self):
        self.background_buildings = []
        for layer in range(1, 4):
            layer_buildings = []
            for i in range(30):
                w = self.rng.integers(50, 150) * (1 / layer)
                h = self.rng.integers(100, self.HEIGHT - 80) * (1 / layer**1.2)
                x = self.rng.integers(-self.WIDTH, self.WORLD_WIDTH + self.WIDTH)
                color_val = 40 - layer * 10
                color = (color_val, color_val + 5, color_val + 20)
                layer_buildings.append({'rect': pygame.Rect(x, self.GROUND_Y - h, w, h), 'color': color, 'layer': layer})
            self.background_buildings.append(layer_buildings)
    
    def _check_collision(self, obj1, obj2):
        # Simple AABB collision for point-like obj1 (projectile)
        return (obj2.x < obj1.x < obj2.x + obj2.width and
                obj2.y < obj1.y < obj2.y + obj2.height)

    def _create_explosion(self, x, y, color, num_particles, max_size):
        for _ in range(num_particles):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.rng.integers(15, 30)
            size = self.rng.integers(2, max_size)
            self.particles.append(Particle(x, y, vx, vy, life, color, size))

    def _get_observation(self):
        # Camera offset for rendering
        cam_offset_x = self.camera_x
        if self.camera_shake > 0:
            cam_offset_x += self.rng.integers(-self.camera_shake, self.camera_shake) / 2

        # --- Render Background ---
        self.screen.fill(self.COLOR_BG)
        for i, layer_buildings in enumerate(self.background_buildings):
            layer = i + 1
            parallax_factor = 1 / (layer * 1.5)
            for building in layer_buildings:
                b_rect = building['rect']
                screen_x = b_rect.x - cam_offset_x * parallax_factor
                if screen_x + b_rect.width > 0 and screen_x < self.WIDTH:
                    pygame.draw.rect(self.screen, building['color'], (screen_x, b_rect.y, b_rect.width, b_rect.height))
        
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

        # --- Render Game Objects ---
        # Enemies
        for enemy in self.enemies:
            e_rect = pygame.Rect(enemy.x - cam_offset_x, enemy.y, enemy.width, enemy.height)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, e_rect, border_radius=4)
            # Health bar
            hp_ratio = enemy.health / enemy.max_health
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (e_rect.left, e_rect.top - 8, enemy.width, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (e_rect.left, e_rect.top - 8, enemy.width * hp_ratio, 5))

        # Player
        p_color = self.COLOR_PLAYER if self.player.invulnerable_timer % 10 < 5 else (200, 200, 255)
        p_rect = pygame.Rect(self.player.x - cam_offset_x, self.player.y, self.player.width, self.player.height)
        pygame.draw.rect(self.screen, p_color, p_rect, border_radius=4)
        # Player "eye"
        eye_x = p_rect.centerx + (5 if self.player.facing_right else -5)
        pygame.draw.circle(self.screen, (255, 255, 255), (eye_x, p_rect.centery - 5), 3)
        
        # Projectiles
        for p in self.projectiles:
            color = self.COLOR_PLAYER_BULLET if p.is_player_bullet else self.COLOR_ENEMY_BULLET
            pos = (int(p.x - cam_offset_x), int(p.y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, color)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = (*p.color, alpha)
            temp_surf = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p.size, p.size), p.size * (p.life / p.max_life))
            self.screen.blit(temp_surf, (p.x - cam_offset_x - p.size, p.y - p.size))

        # --- Render UI ---
        # Health Bar
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (10, 10, 200, 20))
        hp_ratio = max(0, self.player.health / self.player.max_health)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (10, 10, 200 * hp_ratio, 20))
        
        # Ammo
        ammo_text = self.font_ui.render(f"AMMO: {self.player.ammo}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10, 35))

        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Time
        time_left = (self.MAX_STEPS - self.steps) / 60
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, self.HEIGHT - 30))

        # Game Over Message
        if self.game_over:
            msg = "VICTORY!" if self.game_won else "GAME OVER"
            color = self.COLOR_HEALTH_GREEN if self.game_won else self.COLOR_ENEMY
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20, 20))
            self.screen.blit(over_text, text_rect)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player.health,
            "enemies_left": len(self.enemies),
        }
    
    @staticmethod
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

# Example usage:
if __name__ == '__main__':
    # This part is for human play and demonstration
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Change to 'windows' or 'mac' if needed, or remove for default

    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robot Rampage")
    
    running = True
    total_reward = 0
    
    # Mapping keyboard keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        movement_action = 0
        space_action = 0
        shift_action = 0
        
        keys = pygame.key.get_pressed()
        
        # Check movement keys
        if keys[pygame.K_LEFT]:
            movement_action = key_map[pygame.K_LEFT]
        elif keys[pygame.K_RIGHT]:
            movement_action = key_map[pygame.K_RIGHT]
        elif keys[pygame.K_UP]:
            movement_action = key_map[pygame.K_UP]
        
        # Check action keys
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart.")
            
    env.close()
    pygame.quit()