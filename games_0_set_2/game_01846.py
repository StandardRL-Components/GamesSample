
# Generated: 2025-08-27T18:29:28.800471
# Source Brief: brief_01846.md
# Brief Index: 1846

        
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
        "Controls: Use arrow keys to select a turret slot. Hold Shift for Splash or Space for Rapid-Fire turrets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of descending alien invaders by strategically placing turrets."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.Font(None, 48)
        self.font_med = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_BASE = (0, 150, 50)
        self.COLOR_LANDING_ZONE = (40, 50, 60)
        self.COLOR_ALIEN = (220, 50, 50)
        self.COLOR_TURRET_BASIC = (50, 150, 255)
        self.COLOR_TURRET_RAPID = (200, 150, 255)
        self.COLOR_TURRET_SPLASH = (255, 200, 50)
        self.COLOR_PROJECTILE = (100, 255, 100)
        self.COLOR_EXPLOSION = (255, 255, 100)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 50, 50)

        # Game constants
        self.MAX_STEPS = 5000
        self.MAX_ALIENS = 50
        self.BASE_Y_POSITION = self.HEIGHT - 30
        self.LANDING_ZONE_HEIGHT = 10
        self.INITIAL_BASE_HEALTH = 100
        
        self.turret_positions = [
            (self.WIDTH * 0.2, self.HEIGHT * 0.65),
            (self.WIDTH * 0.4, self.HEIGHT * 0.75),
            (self.WIDTH * 0.6, self.HEIGHT * 0.75),
            (self.WIDTH * 0.8, self.HEIGHT * 0.65),
        ]
        
        # Initialize state variables
        self.aliens = []
        self.turrets = []
        self.projectiles = []
        self.explosions = []
        self.occupied_turret_slots = []
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        
        self.aliens.clear()
        self.turrets.clear()
        self.projectiles.clear()
        self.explosions.clear()
        
        self.occupied_turret_slots = [False] * len(self.turret_positions)
        
        self.last_alien_spawn_step = 0
        self._update_difficulty()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # 1. Handle player action
            self._handle_action(action)
            
            # 2. Update game logic
            self._spawn_aliens()
            reward += self._update_turrets()
            reward_from_projectiles = self._update_projectiles()
            reward += reward_from_projectiles
            reward += self._update_aliens()
            self._update_explosions()

            # 3. Update difficulty
            if self.steps > 0 and self.steps % 500 == 0:
                self._update_difficulty()

        # 4. Check for termination
        terminated = self.base_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.base_health <= 0:
                reward -= 100  # Lose penalty
                self.win = False
            else:
                reward += 100  # Win bonus
                self.win = True
        
        # Assertions
        assert self.base_health <= self.INITIAL_BASE_HEALTH
        assert self.score >= 0
        assert len(self.aliens) <= self.MAX_ALIENS

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 0: # No-op
            return

        slot_index = movement - 1 # movement 1-4 maps to index 0-3
        if slot_index < 0 or slot_index >= len(self.turret_positions):
            return

        if self.occupied_turret_slots[slot_index]:
            return # Slot is already occupied

        turret_type = 'basic'
        if shift_held and self.score >= 1500:
            turret_type = 'splash'
        elif space_held and self.score >= 500:
            turret_type = 'rapid'

        turret_config = {
            'basic': {'range': 150, 'fire_rate': 45, 'damage': 1, 'color': self.COLOR_TURRET_BASIC},
            'rapid': {'range': 120, 'fire_rate': 15, 'damage': 1, 'color': self.COLOR_TURRET_RAPID},
            'splash': {'range': 180, 'fire_rate': 60, 'damage': 2, 'color': self.COLOR_TURRET_SPLASH, 'splash_radius': 40}
        }
        config = turret_config[turret_type]

        self.turrets.append({
            'pos': np.array(self.turret_positions[slot_index], dtype=float),
            'type': turret_type,
            'range': config['range'],
            'fire_rate': config['fire_rate'],
            'damage': config['damage'],
            'color': config['color'],
            'last_shot_step': self.steps,
            'angle': -math.pi / 2, # Pointing up initially
            **({'splash_radius': config['splash_radius']} if turret_type == 'splash' else {})
        })
        self.occupied_turret_slots[slot_index] = True
        # sfx: turret_place.wav

    def _update_difficulty(self):
        wave = self.steps // 500
        self.current_wave = wave + 1
        self.alien_spawn_interval = max(15, 60 - wave * 5)
        self.alien_health = 1 + wave
        self.alien_speed = 0.5 + wave * 0.1

    def _spawn_aliens(self):
        if len(self.aliens) < self.MAX_ALIENS and (self.steps - self.last_alien_spawn_step) > self.alien_spawn_interval:
            self.last_alien_spawn_step = self.steps
            spawn_x = self.np_random.uniform(20, self.WIDTH - 20)
            self.aliens.append({
                'pos': np.array([spawn_x, -10.0]),
                'health': self.alien_health,
                'max_health': self.alien_health,
                'speed': self.alien_speed,
                'radius': 8
            })

    def _update_turrets(self):
        for turret in self.turrets:
            # Find closest target
            target_alien = None
            min_dist = float('inf')
            
            for alien in self.aliens:
                dist = np.linalg.norm(turret['pos'] - alien['pos'])
                if dist < turret['range'] and dist < min_dist:
                    min_dist = dist
                    target_alien = alien
            
            # Aim and fire
            if target_alien:
                # Aim
                direction = target_alien['pos'] - turret['pos']
                turret['angle'] = math.atan2(direction[1], direction[0])
                
                # Fire
                if self.steps - turret['last_shot_step'] >= turret['fire_rate']:
                    turret['last_shot_step'] = self.steps
                    self.projectiles.append({
                        'pos': turret['pos'].copy(),
                        'vel': (direction / min_dist) * 5,
                        'type': turret['type'],
                        'damage': turret['damage'],
                        **({'splash_radius': turret['splash_radius']} if turret['type'] == 'splash' else {})
                    })
                    # sfx: laser_shoot.wav
        return 0

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        aliens_hit_this_step = set()

        for i, proj in enumerate(self.projectiles):
            proj['pos'] += proj['vel']

            # Check for off-screen
            if not (0 <= proj['pos'][0] < self.WIDTH and 0 <= proj['pos'][1] < self.HEIGHT):
                projectiles_to_remove.append(i)
                reward -= 0.1 # Missed shot penalty
                continue

            # Check for collision
            hit = False
            for j, alien in enumerate(self.aliens):
                if np.linalg.norm(proj['pos'] - alien['pos']) < alien['radius']:
                    hit = True
                    projectiles_to_remove.append(i)
                    
                    if proj['type'] == 'splash':
                        # sfx: splash_explosion.wav
                        self.explosions.append({'pos': proj['pos'], 'radius': 10, 'max_radius': proj['splash_radius'], 'duration': 15, 'lived': 0})
                        # Splash damage affects all aliens in radius
                        for k, other_alien in enumerate(self.aliens):
                            if np.linalg.norm(proj['pos'] - other_alien['pos']) < proj['splash_radius']:
                                other_alien['health'] -= proj['damage']
                                aliens_hit_this_step.add(k)
                    else:
                        # sfx: alien_hit.wav
                        alien['health'] -= proj['damage']
                        aliens_hit_this_step.add(j)
                        self.explosions.append({'pos': alien['pos'], 'radius': 5, 'max_radius': 15, 'duration': 5, 'lived': 0})
                    
                    break # Projectile can only hit one alien directly
        
        reward += 1.0 * len(aliens_hit_this_step)

        # Remove projectiles that hit or went off-screen
        for i in sorted(projectiles_to_remove, reverse=True):
            del self.projectiles[i]
            
        return reward

    def _update_aliens(self):
        reward = 0
        aliens_to_remove = []
        for i, alien in enumerate(self.aliens):
            # Health check
            if alien['health'] <= 0:
                if i not in aliens_to_remove:
                    aliens_to_remove.append(i)
                    reward += 10 # Destroyed alien bonus
                    self.score += 10
                    # sfx: alien_destroy.wav
                    self.explosions.append({'pos': alien['pos'], 'radius': 10, 'max_radius': 30, 'duration': 10, 'lived': 0})
                continue
                
            # Movement
            alien['pos'][1] += alien['speed']
            
            # Check for reaching base
            if alien['pos'][1] > self.BASE_Y_POSITION:
                if i not in aliens_to_remove:
                    aliens_to_remove.append(i)
                    self.base_health -= 10
                    reward -= 5 # Reached base penalty
                    self.score = max(0, self.score - 5)
                    # sfx: base_hit.wav
                    self.explosions.append({'pos': (self.WIDTH/2, self.BASE_Y_POSITION), 'radius': 20, 'max_radius': 80, 'duration': 20, 'lived': 0, 'color': (200, 80, 80)})

        # Remove destroyed or landed aliens
        for i in sorted(aliens_to_remove, reverse=True):
            del self.aliens[i]
            
        return reward

    def _update_explosions(self):
        explosions_to_remove = []
        for i, exp in enumerate(self.explosions):
            exp['lived'] += 1
            exp['radius'] = exp['max_radius'] * math.sin((exp['lived'] / exp['duration']) * math.pi/2)
            if exp['lived'] >= exp['duration']:
                explosions_to_remove.append(i)
        
        for i in sorted(explosions_to_remove, reverse=True):
            del self.explosions[i]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw base and landing zone
        pygame.draw.rect(self.screen, self.COLOR_LANDING_ZONE, (0, self.BASE_Y_POSITION, self.WIDTH, self.LANDING_ZONE_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.WIDTH/2 - 50, self.BASE_Y_POSITION, 100, self.LANDING_ZONE_HEIGHT))

        # Draw turret placement slots
        for i, pos in enumerate(self.turret_positions):
            if not self.occupied_turret_slots[i]:
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 15, (100, 100, 100, 100))

        # Draw turrets
        for turret in self.turrets:
            p1 = (turret['pos'][0] + 15 * math.cos(turret['angle']), turret['pos'][1] + 15 * math.sin(turret['angle']))
            p2 = (turret['pos'][0] + 10 * math.cos(turret['angle'] + 2.356), turret['pos'][1] + 10 * math.sin(turret['angle'] + 2.356))
            p3 = (turret['pos'][0] + 10 * math.cos(turret['angle'] - 2.356), turret['pos'][1] + 10 * math.sin(turret['angle'] - 2.356))
            points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
            pygame.gfxdraw.aapolygon(self.screen, points, turret['color'])
            pygame.gfxdraw.filled_polygon(self.screen, points, turret['color'])

        # Draw aliens and their health bars
        for alien in self.aliens:
            pos = (int(alien['pos'][0]), int(alien['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], alien['radius'], self.COLOR_ALIEN)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], alien['radius'], self.COLOR_ALIEN)
            # Health bar
            if alien['health'] < alien['max_health']:
                bar_width = 20
                health_pct = alien['health'] / alien['max_health']
                pygame.draw.rect(self.screen, (80,0,0), (pos[0] - bar_width/2, pos[1] - 15, bar_width, 3))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (pos[0] - bar_width/2, pos[1] - 15, bar_width * health_pct, 3))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, (pos[0]-1, pos[1]-1, 3, 3))

        # Draw explosions
        for exp in self.explosions:
            color = exp.get('color', self.COLOR_EXPLOSION)
            alpha = int(255 * (1 - (exp['lived'] / exp['duration'])))
            color_with_alpha = (*color, alpha)
            pygame.gfxdraw.aacircle(self.screen, int(exp['pos'][0]), int(exp['pos'][1]), int(exp['radius']), color_with_alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(exp['pos'][0]), int(exp['pos'][1]), int(exp['radius']), color_with_alpha)

    def _render_ui(self):
        # Score
        score_text = self.font_med.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Base Health
        health_text = self.font_med.render("BASE HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (self.WIDTH - health_text.get_width() - 10, 10))
        health_pct = max(0, self.base_health / self.INITIAL_BASE_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (self.WIDTH - 160, 40, 150, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (self.WIDTH - 160, 40, 150 * health_pct, 20))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH/2 - wave_text.get_width()/2, self.HEIGHT - 25))

        # Turret Unlocks
        if self.score < 500:
            unlock_text = self.font_small.render("Rapid-Fire [SPACE] unlocks at 500", True, (150,150,150))
            self.screen.blit(unlock_text, (10, 40))
        if self.score < 1500:
            unlock_text = self.font_small.render("Splash [SHIFT] unlocks at 1500", True, (150,150,150))
            self.screen.blit(unlock_text, (10, 60))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "VICTORY" if self.win else "GAME OVER"
            end_text = self.font_big.render(end_text_str, True, self.COLOR_EXPLOSION if self.win else self.COLOR_ALIEN)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.font_med.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 30))
            self.screen.blit(final_score_text, score_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "wave": self.current_wave,
            "aliens_on_screen": len(self.aliens)
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # This is a basic manual player loop.
    # It demonstrates how actions are mapped.
    
    # Set render_mode to 'human' if you want to see the window
    # Note: The provided class is set up for 'rgb_array' for headless operation.
    # To see the game, you'd need to modify the __init__ and _get_observation.
    # For this example, we'll just print states and use a dummy loop.
    
    # For demonstration, we'll use a simple "AI" that places a turret and then waits.
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Let's run a short game episode
    for step in range(env.MAX_STEPS + 10):
        if done:
            break
            
        # Simple agent: Place a basic turret in slot 1 at the beginning
        if step == 10:
            action = np.array([1, 0, 0]) # Move=1 (slot 0), no space, no shift
        else:
            action = env.action_space.sample() # Random actions
            action[0] = 0 # Mostly do nothing to let turrets work
            if random.random() < 0.05: # Occasionally try to build
                action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # To see the output, we need a way to render.
        # Let's save a few frames to disk to see what's happening.
        if step % 100 == 0:
            # The observation is (H, W, C). Pygame surface wants (W, H).
            # So we need to transpose back.
            img_array = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(img_array)
            pygame.image.save(surf, f"frame_{step:04d}.png")
            print(f"Step: {step}, Info: {info}, Reward: {reward:.2f}")

    print(f"Episode finished. Final Info: {info}, Total Reward: {total_reward:.2f}")
    env.close()