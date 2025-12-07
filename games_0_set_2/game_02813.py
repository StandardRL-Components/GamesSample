
# Generated: 2025-08-28T06:04:56.527130
# Source Brief: brief_02813.md
# Brief Index: 2813

        
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
        "Controls: Arrow keys to move. Hold space to fire. Collect power-ups!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a powerful robot in a top-down arena, blasting enemy robots "
        "while collecting power-ups to achieve total robotic domination."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.WIN_CONDITION = 15
        
        # Colors
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_ARENA = (20, 30, 40)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 150)
        self.COLOR_ENEMY = (255, 50, 100)
        self.COLOR_PLAYER_PROJ = (255, 255, 100)
        self.COLOR_ENEMY_PROJ = (255, 150, 0)
        self.COLOR_HEALTH_POWERUP = (0, 255, 120)
        self.COLOR_DAMAGE_POWERUP = (200, 100, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (40, 200, 40)
        self.COLOR_HEALTH_BAR_BG = (80, 40, 40)
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game parameters
        self.player_size = 20
        self.player_speed = 4
        self.player_fire_rate = 8 # steps between shots
        self.player_max_health = 100
        self.player_proj_speed = 8
        
        self.enemy_size = 22
        self.enemy_speed = 1.5
        self.enemy_base_health = 3
        self.enemy_fire_rate = 50 # steps between shots
        
        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_last_fire = None
        self.player_damage_boost_timer = None
        self.last_move_dir = None
        
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.powerups = []
        self.particles = []
        
        self.enemy_spawn_timer = None
        self.enemy_spawn_rate = None
        self.enemy_proj_speed = None
        
        self.steps = 0
        self.score = 0
        self.enemies_defeated = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.player_max_health
        self.player_last_fire = 0
        self.player_damage_boost_timer = 0
        self.last_move_dir = pygame.Vector2(0, -1) # Default fire upwards

        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.powerups = []
        self.particles = []
        
        self.enemy_spawn_timer = 0
        self.enemy_spawn_rate = 100 # Initial spawn rate
        self.enemy_proj_speed = 2.0 # Initial projectile speed

        self.steps = 0
        self.score = 0
        self.enemies_defeated = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            self.steps += 1
            
            # 1. Handle player input
            self._handle_player_input(action)
            
            # 2. Update game state
            self._update_difficulty()
            self._update_spawners()
            self._update_entities()
            
            # 3. Handle collisions and calculate rewards
            reward += self._handle_collisions()
            
            # 4. Check for termination
            terminated, terminal_reward = self._check_termination()
            reward += terminal_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, action):
        movement, space_held, _ = action
        
        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right
        
        if move_vec.length() > 0:
            self.last_move_dir = move_vec.normalize()
            self.player_pos += self.last_move_dir * self.player_speed
        
        # Clamp player position to stay within arena
        self.player_pos.x = np.clip(self.player_pos.x, self.player_size / 2, self.WIDTH - self.player_size / 2)
        self.player_pos.y = np.clip(self.player_pos.y, self.player_size / 2, self.HEIGHT - self.player_size / 2)
        
        # Firing
        if space_held and (self.steps - self.player_last_fire) > self.player_fire_rate:
            self.player_last_fire = self.steps
            proj_pos = self.player_pos + self.last_move_dir * (self.player_size / 2)
            proj_vel = self.last_move_dir * self.player_proj_speed
            self.player_projectiles.append({'pos': proj_pos, 'vel': proj_vel})
            # sfx: player_shoot.wav

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 500 == 0:
            self.enemy_spawn_rate = max(30, self.enemy_spawn_rate - 10)
        if self.steps > 0 and self.steps % 250 == 0:
            self.enemy_proj_speed = min(5.0, self.enemy_proj_speed + 0.1)

    def _update_spawners(self):
        self.enemy_spawn_timer += 1
        if self.enemy_spawn_timer >= self.enemy_spawn_rate:
            self.enemy_spawn_timer = 0
            side = self.np_random.integers(4)
            if side == 0: pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -self.enemy_size)
            elif side == 1: pos = pygame.Vector2(self.WIDTH + self.enemy_size, self.np_random.uniform(0, self.HEIGHT))
            elif side == 2: pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.enemy_size)
            else: pos = pygame.Vector2(-self.enemy_size, self.np_random.uniform(0, self.HEIGHT))
            
            self.enemies.append({
                'pos': pos, 
                'health': self.enemy_base_health, 
                'fire_timer': self.np_random.integers(0, self.enemy_fire_rate)
            })

    def _update_entities(self):
        # Update player boost timer
        if self.player_damage_boost_timer > 0:
            self.player_damage_boost_timer -= 1

        # Update enemies
        for enemy in self.enemies:
            direction = (self.player_pos - enemy['pos']).normalize() if (self.player_pos - enemy['pos']).length() > 0 else pygame.Vector2(0,0)
            enemy['pos'] += direction * self.enemy_speed
            enemy['fire_timer'] += 1
            if enemy['fire_timer'] >= self.enemy_fire_rate:
                enemy['fire_timer'] = 0
                proj_pos = enemy['pos'] + direction * (self.enemy_size / 2)
                proj_vel = direction * self.enemy_proj_speed
                self.enemy_projectiles.append({'pos': proj_pos, 'vel': proj_vel})
                # sfx: enemy_shoot.wav

        # Update projectiles
        self.player_projectiles = [p for p in self.player_projectiles if self._update_projectile(p)]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if self._update_projectile(p)]

        # Update particles
        self.particles = [p for p in self.particles if self._update_particle(p)]

    def _update_projectile(self, p):
        p['pos'] += p['vel']
        return 0 < p['pos'].x < self.WIDTH and 0 < p['pos'].y < self.HEIGHT

    def _update_particle(self, p):
        p['pos'] += p['vel']
        p['life'] -= 1
        return p['life'] > 0

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if (proj['pos'] - enemy['pos']).length() < (self.enemy_size / 2):
                    damage = 2 if self.player_damage_boost_timer > 0 else 1
                    enemy['health'] -= damage
                    reward += 0.1
                    self.player_projectiles.remove(proj)
                    if enemy['health'] <= 0:
                        reward += 5
                        self.score += 100
                        self.enemies_defeated += 1
                        self._create_explosion(enemy['pos'], self.COLOR_ENEMY, 30)
                        self.enemies.remove(enemy)
                        # sfx: explosion.wav
                        if self.np_random.random() < 0.2: # 20% chance to drop powerup
                            ptype = 'health' if self.np_random.random() < 0.5 else 'damage'
                            self.powerups.append({'pos': enemy['pos'].copy(), 'type': ptype})
                    break
        
        # Enemy projectiles vs Player
        player_rect = pygame.Rect(self.player_pos.x - self.player_size/2, self.player_pos.y - self.player_size/2, self.player_size, self.player_size)
        for proj in self.enemy_projectiles[:]:
            if player_rect.collidepoint(proj['pos']):
                self.player_health -= 10
                reward -= 0.1
                self.enemy_projectiles.remove(proj)
                self._create_explosion(self.player_pos, self.COLOR_PLAYER_GLOW, 10)
                # sfx: player_hit.wav
                break
        
        # Player vs Powerups
        for powerup in self.powerups[:]:
            if (self.player_pos - powerup['pos']).length() < self.player_size:
                reward += 1
                self.score += 50
                if powerup['type'] == 'health':
                    self.player_health = min(self.player_max_health, self.player_health + 25)
                    # sfx: powerup_health.wav
                elif powerup['type'] == 'damage':
                    self.player_damage_boost_timer = 300 # 10 seconds at 30fps
                    # sfx: powerup_damage.wav
                self.powerups.remove(powerup)
        
        return reward

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            self._create_explosion(self.player_pos, self.COLOR_PLAYER, 50)
            return True, -100 # Defeat
        if self.enemies_defeated >= self.WIN_CONDITION:
            self.game_over = True
            return True, 100 # Victory
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True, 0 # Timeout
        return False, 0

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'color': color})
            
    def _get_observation(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (20, 20, self.WIDTH - 40, self.HEIGHT - 40))

        # Render powerups (underneath other entities)
        for powerup in self.powerups:
            pulse = abs(math.sin(self.steps * 0.1))
            radius = int(10 + pulse * 4)
            color = self.COLOR_HEALTH_POWERUP if powerup['type'] == 'health' else self.COLOR_DAMAGE_POWERUP
            pygame.gfxdraw.filled_circle(self.screen, int(powerup['pos'].x), int(powerup['pos'].y), radius, color)
            pygame.gfxdraw.aacircle(self.screen, int(powerup['pos'].x), int(powerup['pos'].y), radius, color)

        # Render projectiles
        for p in self.player_projectiles:
            end_pos = p['pos'] - p['vel'].normalize() * 5
            pygame.draw.aaline(self.screen, self.COLOR_PLAYER_PROJ, p['pos'], end_pos, 2)
        for p in self.enemy_projectiles:
            pygame.draw.aaline(self.screen, self.COLOR_ENEMY_PROJ, p['pos'], (p['pos'].x, p['pos'].y), 2)
            
        # Render enemies
        for enemy in self.enemies:
            rect = pygame.Rect(0, 0, self.enemy_size, self.enemy_size)
            rect.center = (int(enemy['pos'].x), int(enemy['pos'].y))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=4)
            
        # Render player
        player_rect = pygame.Rect(0, 0, self.player_size, self.player_size)
        player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Glow effect
        glow_size = int(self.player_size * (1.2 + 0.2 * abs(math.sin(self.steps * 0.2))))
        if self.player_damage_boost_timer > 0:
            glow_color = self.COLOR_DAMAGE_POWERUP
            glow_size = int(self.player_size * (1.5 + 0.3 * abs(math.sin(self.steps * 0.3))))
        else:
            glow_color = self.COLOR_PLAYER_GLOW
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), glow_size, glow_color)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 4, 4))
            self.screen.blit(temp_surf, p['pos'] - pygame.Vector2(2, 2))

        # Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.player_max_health)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, 20))
        
        # Score and Enemies defeated
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        enemies_text = self.font_small.render(f"DEFEATED: {self.enemies_defeated} / {self.WIN_CONDITION}", True, self.COLOR_UI_TEXT)
        self.screen.blit(enemies_text, (self.WIDTH - enemies_text.get_width() - 10, 35))

        # Game Over / Victory Message
        if self.game_over:
            if self.enemies_defeated >= self.WIN_CONDITION:
                msg = "VICTORY"
                color = self.COLOR_HEALTH_POWERUP
            else:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "enemies_defeated": self.enemies_defeated,
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robot Arena")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Mapping from Pygame keys to your action space
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
        
        # Check movement keys
        for key, action_val in key_to_action.items():
            if keys[key]:
                movement_action = action_val
                break # Prioritize first key found (e.g., up over down)

        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()