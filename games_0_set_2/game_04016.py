
# Generated: 2025-08-28T01:08:23.596975
# Source Brief: brief_04016.md
# Brief Index: 4016

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend Earth from a wave of descending alien invaders in this top-down, "
        "procedurally generated arcade shooter."
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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_PROJ = (0, 200, 255)
        self.COLOR_ENEMY_PROJ = (255, 50, 50)
        self.COLOR_ALIEN_RED = (255, 80, 80)
        self.COLOR_ALIEN_BLUE = (100, 100, 255)
        self.COLOR_EXPLOSION = (255, 200, 0)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_HEALTH_BAR = (0, 255, 0)
        self.COLOR_HEALTH_BAR_BG = (50, 50, 50)

        # Game constants
        self.MAX_STEPS = 5000
        self.TOTAL_ALIENS = 50
        
        self.PLAYER_SPEED = 5
        self.PLAYER_MAX_HEALTH = 50
        self.PLAYER_FIRE_COOLDOWN = 6 # frames
        self.PLAYER_PROJ_SPEED = 8
        self.PLAYER_HITBOX_RADIUS = 8
        
        self.ALIEN_PROJ_DAMAGE = 1
        self.ALIEN_HITBOX_RADIUS = 10
        self.NEAR_MISS_RADIUS = 40
        
        # Initialize state variables (to be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_health = 0
        self.player_fire_timer = 0
        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.explosions = []
        self.alien_base_fire_rate_factor = 1.0
        self.alien_projectile_speed = 3.0
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 40]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_fire_timer = 0
        
        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.explosions = []
        
        self.alien_base_fire_rate_factor = 1.0
        self.alien_projectile_speed = 3.0
        
        self._spawn_aliens()

        return self._get_observation(), self._get_info()

    def _spawn_aliens(self):
        for _ in range(self.TOTAL_ALIENS):
            alien_type = self.np_random.choice(['red', 'blue'])
            cooldown_time = 20 if alien_type == 'red' else 30
            
            self.aliens.append({
                'pos': [self.np_random.integers(20, self.WIDTH - 20), self.np_random.integers(20, self.HEIGHT // 2)],
                'type': alien_type,
                'cooldown': self.np_random.integers(0, cooldown_time),
                'descent_speed': self.np_random.uniform(0.1, 0.3)
            })
    
    def step(self, action):
        reward = -0.01  # Penalty for time passing
        self.game_over = self._check_termination()
        if self.game_over:
            if self.player_health <= 0: reward -= 100
            elif len(self.aliens) == 0: reward += 100
            return self._get_observation(), reward, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- LOGIC UPDATES ---
        
        # 1. Update timers
        self.steps += 1
        if self.player_fire_timer > 0: self.player_fire_timer -= 1
        
        # 2. Handle player input and movement
        closest_alien_dist_before = self._get_closest_alien_dist()
        
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)
        
        closest_alien_dist_after = self._get_closest_alien_dist()
        if closest_alien_dist_after < closest_alien_dist_before:
            reward += 0.1

        # 3. Handle player firing
        if space_held and self.player_fire_timer == 0:
            # sfx: player_shoot.wav
            self.player_projectiles.append({
                'pos': self.player_pos.copy(),
                'near_missed': set()
            })
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

        # 4. Update aliens
        for alien in self.aliens:
            alien['pos'][1] += alien['descent_speed']
            alien['cooldown'] -= 1
            if alien['cooldown'] <= 0:
                self._alien_fire(alien)

        # 5. Update projectiles
        self._update_projectiles()

        # 6. Handle collisions and rewards
        reward += self._handle_collisions()

        # 7. Update explosions and other effects
        self._update_explosions()

        # 8. Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.alien_base_fire_rate_factor = max(0.5, self.alien_base_fire_rate_factor * 0.99)
        if self.steps > 0 and self.steps % 500 == 0:
            self.alien_projectile_speed = min(6.0, self.alien_projectile_speed + 0.1)

        # 9. Check termination state
        terminated = self._check_termination()
        if terminated:
            if self.player_health <= 0: reward -= 100
            elif len(self.aliens) == 0: reward += 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_closest_alien_dist(self):
        if not self.aliens:
            return float('inf')
        player_p = np.array(self.player_pos)
        alien_positions = np.array([a['pos'] for a in self.aliens])
        distances = np.linalg.norm(alien_positions - player_p, axis=1)
        return np.min(distances)

    def _alien_fire(self, alien):
        # sfx: enemy_shoot.wav
        if alien['type'] == 'red':
            direction = np.array(self.player_pos) - np.array(alien['pos'])
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
                self.enemy_projectiles.append({
                    'pos': alien['pos'].copy(),
                    'vel': direction * self.alien_projectile_speed
                })
            alien['cooldown'] = int(20 * self.alien_base_fire_rate_factor)

        elif alien['type'] == 'blue':
            for i in range(3):
                angle_offset = math.radians((i - 1) * 15) # -15, 0, +15 degrees
                direction = np.array(self.player_pos) - np.array(alien['pos'])
                norm = np.linalg.norm(direction)
                if norm > 0:
                    base_angle = math.atan2(direction[1], direction[0])
                    final_angle = base_angle + angle_offset
                    vel = np.array([math.cos(final_angle), math.sin(final_angle)]) * self.alien_projectile_speed
                    self.enemy_projectiles.append({
                        'pos': alien['pos'].copy(),
                        'vel': vel
                    })
            alien['cooldown'] = int(30 * self.alien_base_fire_rate_factor)

    def _update_projectiles(self):
        # Player projectiles move up
        self.player_projectiles = [p for p in self.player_projectiles if p['pos'][1] > 0]
        for p in self.player_projectiles:
            p['pos'][1] -= self.PLAYER_PROJ_SPEED
            
        # Enemy projectiles move based on velocity
        self.enemy_projectiles = [
            p for p in self.enemy_projectiles 
            if 0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT
        ]
        for p in self.enemy_projectiles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs aliens
        projectiles_to_remove = []
        aliens_to_remove = []
        for i, p in enumerate(self.player_projectiles):
            for j, a in enumerate(self.aliens):
                if j in aliens_to_remove: continue
                dist = math.hypot(p['pos'][0] - a['pos'][0], p['pos'][1] - a['pos'][1])
                
                # Hit detection
                if dist < self.ALIEN_HITBOX_RADIUS:
                    # sfx: explosion.wav
                    self.explosions.append({'pos': a['pos'].copy(), 'radius': 30, 'life': 15})
                    aliens_to_remove.append(j)
                    projectiles_to_remove.append(i)
                    reward += 10
                    self.score += 1
                    break
                # Near miss detection
                elif dist < self.NEAR_MISS_RADIUS and j not in p['near_missed']:
                    reward += 2
                    p['near_missed'].add(j)

        # Remove collided entities (in reverse to avoid index errors)
        for i in sorted(list(set(projectiles_to_remove)), reverse=True):
            del self.player_projectiles[i]
        for i in sorted(list(set(aliens_to_remove)), reverse=True):
            del self.aliens[i]
            
        # Enemy projectiles vs player
        projectiles_to_remove = []
        for i, p in enumerate(self.enemy_projectiles):
            dist = math.hypot(p['pos'][0] - self.player_pos[0], p['pos'][1] - self.player_pos[1])
            if dist < self.PLAYER_HITBOX_RADIUS:
                # sfx: player_hit.wav
                self.player_health = max(0, self.player_health - self.ALIEN_PROJ_DAMAGE)
                projectiles_to_remove.append(i)
                reward -= 1
                self.explosions.append({'pos': self.player_pos.copy(), 'radius': 15, 'life': 10})
        
        for i in sorted(projectiles_to_remove, reverse=True):
            del self.enemy_projectiles[i]
            
        return reward

    def _update_explosions(self):
        self.explosions = [e for e in self.explosions if e['life'] > 0]
        for e in self.explosions:
            e['life'] -= 1

    def _check_termination(self):
        if self.player_health <= 0: return True
        if not self.aliens: return True
        if self.steps >= self.MAX_STEPS: return True
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
            "health": self.player_health,
            "aliens_remaining": len(self.aliens)
        }

    def _render_game(self):
        # Render aliens
        for alien in self.aliens:
            color = self.COLOR_ALIEN_RED if alien['type'] == 'red' else self.COLOR_ALIEN_BLUE
            pos = (int(alien['pos'][0]), int(alien['pos'][1]))
            size = self.ALIEN_HITBOX_RADIUS
            pygame.draw.rect(self.screen, color, (pos[0] - size, pos[1] - size, 2 * size, 2 * size))

        # Render player projectiles
        for p in self.player_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PLAYER_PROJ)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PLAYER_PROJ)

        # Render enemy projectiles
        for p in self.enemy_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_ENEMY_PROJ)

        # Render player
        if self.player_health > 0:
            p1 = (self.player_pos[0], self.player_pos[1] - 12)
            p2 = (self.player_pos[0] - 8, self.player_pos[1] + 8)
            p3 = (self.player_pos[0] + 8, self.player_pos[1] + 8)
            points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Render explosions
        for e in self.explosions:
            alpha = int(255 * (e['life'] / 15.0))
            color = (*self.COLOR_EXPLOSION, alpha)
            temp_surf = pygame.Surface((e['radius']*2, e['radius']*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, e['radius'], e['radius'], e['radius'], color)
            self.screen.blit(temp_surf, (int(e['pos'][0] - e['radius']), int(e['pos'][1] - e['radius'])))

    def _render_ui(self):
        # Render score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Render health bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 150
        current_health_width = int(bar_width * health_ratio)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        if current_health_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, current_health_width, 20))
        
        # Render Game Over/Win message
        if self.game_over:
            if self.player_health <= 0:
                msg = "GAME OVER"
            else:
                msg = "YOU WIN!"
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_game_over.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

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
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Test a few random steps
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"Episode finished after {_+1} steps. Score: {info['score']}")
            obs, info = env.reset()
    
    env.close()
    
    # Example with rendering to a window for visualization
    # To run this, you might need to comment out the `os.environ` line above
    # and install pygame properly.
    try:
        del os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        pass

    print("\nStarting interactive visualization...")
    env_vis = GameEnv(render_mode="rgb_array")
    obs, info = env_vis.reset()
    
    # Pygame window setup for visualization
    vis_screen = pygame.display.set_mode((env_vis.WIDTH, env_vis.HEIGHT))
    pygame.display.set_caption("Arcade Shooter")
    running = True
    
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env_vis.step(action)
        
        # Blit the observation onto the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        vis_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}. Resetting in 3 seconds...")
            pygame.time.wait(3000)
            env_vis.reset()
            
        env_vis.clock.tick(30) # Limit to 30 FPS
        
    env_vis.close()