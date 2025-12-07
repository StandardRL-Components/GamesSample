
# Generated: 2025-08-28T02:39:46.986954
# Source Brief: brief_01769.md
# Brief Index: 1769

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A top-down arcade shooter where the player must survive three waves of increasingly difficult alien invaders.
    The player controls a spaceship, dodging enemy fire and destroying aliens.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ↑↓←→ to move. Hold space to fire. Hold shift for a temporary shield."
    )

    # User-facing description of the game
    game_description = (
        "Survive three waves of alien invaders in this fast-paced arcade shooter. "
        "Destroy all aliens to win, but watch your health!"
    )

    # Frames auto-advance at 30fps
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.width, self.height = 640, 400

        # EXACT spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game constants
        self.MAX_STEPS = 3600  # 2 minutes at 30fps

        self.PLAYER_SPEED = 6
        self.PLAYER_FIRE_COOLDOWN = 6 # 5 shots per second
        self.PLAYER_HEALTH_MAX = 100
        
        self.SHIELD_DURATION = 90 # 3 seconds
        self.SHIELD_COOLDOWN = 180 # 6 seconds

        self.PROJECTILE_SPEED_PLAYER = 10
        self.PROJECTILE_SPEED_ENEMY_BASE = 4

        self.WAVE_TRANSITION_TIME = 90 # 3 seconds

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_SHIELD = (100, 180, 255, 100)
        self.COLOR_ALIEN_BASIC = (255, 80, 80)
        self.COLOR_ALIEN_ADVANCED = (255, 80, 200)
        self.COLOR_ALIEN_BOMBER = (255, 160, 50)
        self.COLOR_PROJECTILE_PLAYER = (255, 255, 100)
        self.COLOR_PROJECTILE_ENEMY = (255, 100, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_GREEN = (50, 200, 50)
        self.COLOR_HEALTH_RED = (200, 50, 50)

        # Initialize state variables
        self.np_random = None
        self.player_pos = None
        self.player_health = None
        self.player_fire_cooldown_timer = None
        self.player_hit_flash_timer = None
        self.shield_active = None
        self.shield_timer = None
        self.shield_cooldown_timer = None
        self.aliens = None
        self.player_projectiles = None
        self.enemy_projectiles = None
        self.particles = None
        self.stars = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.victory = None
        self.current_wave = None
        self.wave_transition_timer = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.width / 2, self.height - 50)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_fire_cooldown_timer = 0
        self.player_hit_flash_timer = 0
        
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0

        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = [
            (random.randint(0, self.width), random.randint(0, self.height), random.randint(1, 3))
            for _ in range(150)
        ]
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.current_wave = 0
        self.wave_transition_timer = self.WAVE_TRANSITION_TIME

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False
        
        self.steps += 1
        
        # Handle game over or victory states
        if self.game_over or self.victory:
            return self._get_observation(), 0, True, False, self._get_info()

        # Handle wave transitions
        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1
            if self.wave_transition_timer == 0:
                self._start_next_wave()
        else:
            # Main game logic
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            reward += 0.01 # Small survival reward
            
            self._handle_player_input(movement, space_held, shift_held)
            reward += self._update_shield(shift_held)
            
            self._update_player_projectiles()
            self._update_enemy_projectiles()
            self._update_aliens()

            collision_rewards = self._handle_collisions()
            reward += collision_rewards
        
        self._update_particles()
        self._update_timers()
        
        # Check for wave completion
        if len(self.aliens) == 0 and self.current_wave > 0 and not self.game_over and not self.victory:
            if self.current_wave == 3:
                self.victory = True
                reward += 100
            else:
                self.wave_transition_timer = self.WAVE_TRANSITION_TIME
                reward += 50
        
        # Check termination conditions
        if self.player_health <= 0:
            self.game_over = True
            reward -= 50
            self._create_explosion(self.player_pos, 40, (255, 100, 100))
        
        if self.steps >= self.MAX_STEPS and not self.victory:
            self.game_over = True
        
        terminated = self.game_over or self.victory

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        enemy_projectile_speed = self.PROJECTILE_SPEED_ENEMY_BASE + (self.current_wave - 1) * 0.5

        wave_configs = {
            1: {'basic': 5, 'advanced': 0, 'bomber': 0},
            2: {'basic': 5, 'advanced': 2, 'bomber': 0},
            3: {'basic': 3, 'advanced': 4, 'bomber': 2},
        }
        config = wave_configs[self.current_wave]
        num_aliens = sum(config.values())
        
        for i in range(num_aliens):
            alien_type = ''
            if i < config['basic']:
                alien_type = 'basic'
            elif i < config['basic'] + config['advanced']:
                alien_type = 'advanced'
            else:
                alien_type = 'bomber'
            
            pos_x = (self.width / (num_aliens + 1)) * (i + 1)
            pos_y = 60 + (i % 2) * 40
            
            self.aliens.append({
                'pos': pygame.Vector2(pos_x, pos_y),
                'type': alien_type,
                'health': 20,
                'fire_cooldown': random.randint(60, 120),
                'move_pattern': {'offset': random.uniform(0, 2 * math.pi), 'speed': random.uniform(0.02, 0.04)},
                'projectile_speed': enemy_projectile_speed
            })

    def _handle_player_input(self, movement, space_held, shift_held):
        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED
        if movement == 2: self.player_pos.y += self.PLAYER_SPEED
        if movement == 3: self.player_pos.x -= self.PLAYER_SPEED
        if movement == 4: self.player_pos.x += self.PLAYER_SPEED

        self.player_pos.x = np.clip(self.player_pos.x, 10, self.width - 10)
        self.player_pos.y = np.clip(self.player_pos.y, 10, self.height - 10)

        if space_held and self.player_fire_cooldown_timer == 0:
            # Sound: player_shoot
            self.player_projectiles.append(pygame.Vector2(self.player_pos))
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_shield(self, shift_held):
        reward = 0
        if shift_held and not self.shield_active and self.shield_cooldown_timer == 0:
            self.shield_active = True
            self.shield_timer = self.SHIELD_DURATION
            self.shield_cooldown_timer = self.SHIELD_COOLDOWN
            # Sound: shield_up
            
            # Check for unnecessary use
            is_necessary = False
            for proj in self.enemy_projectiles:
                if proj.distance_to(self.player_pos) < 150:
                    is_necessary = True
                    break
            if not is_necessary:
                reward -= 0.2
        return reward

    def _update_player_projectiles(self):
        for proj in self.player_projectiles[:]:
            proj.y -= self.PROJECTILE_SPEED_PLAYER
            if proj.y < 0:
                self.player_projectiles.remove(proj)

    def _update_enemy_projectiles(self):
        for proj in self.enemy_projectiles[:]:
            proj.y += proj.speed
            if proj.y > self.height:
                self.enemy_projectiles.remove(proj)
    
    def _update_aliens(self):
        for alien in self.aliens:
            pattern = alien['move_pattern']
            alien['pos'].x += math.sin(self.steps * pattern['speed'] + pattern['offset']) * 2
            
            alien['fire_cooldown'] -= 1
            if alien['fire_cooldown'] <= 0:
                # Sound: enemy_shoot
                base_cooldown = 120
                if alien['type'] == 'basic':
                    proj = pygame.Vector2(alien['pos'])
                    proj.speed = alien['projectile_speed']
                    self.enemy_projectiles.append(proj)
                    alien['fire_cooldown'] = random.randint(base_cooldown, base_cooldown + 60)
                elif alien['type'] == 'advanced':
                    for offset in [-5, 5]:
                        proj = pygame.Vector2(alien['pos'].x + offset, alien['pos'].y)
                        proj.speed = alien['projectile_speed']
                        self.enemy_projectiles.append(proj)
                    alien['fire_cooldown'] = random.randint(base_cooldown - 30, base_cooldown + 30)
                elif alien['type'] == 'bomber':
                    proj = pygame.Vector2(alien['pos'])
                    proj.speed = alien['projectile_speed'] * 0.8
                    proj.is_bomber = True
                    self.enemy_projectiles.append(proj)
                    alien['fire_cooldown'] = random.randint(base_cooldown + 60, base_cooldown + 120)

    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if proj.distance_to(alien['pos']) < 15:
                    # Sound: alien_hit
                    self.player_projectiles.remove(proj)
                    alien['health'] -= 10
                    if alien['health'] <= 0:
                        reward_map = {'basic': 1, 'advanced': 2, 'bomber': 3}
                        score_map = {'basic': 100, 'advanced': 200, 'bomber': 300}
                        color_map = {'basic': self.COLOR_ALIEN_BASIC, 'advanced': self.COLOR_ALIEN_ADVANCED, 'bomber': self.COLOR_ALIEN_BOMBER}
                        
                        reward += reward_map[alien['type']]
                        self.score += score_map[alien['type']]
                        self._create_explosion(alien['pos'], 20, color_map[alien['type']])
                        self.aliens.remove(alien)
                    break
        
        # Enemy projectiles vs Player
        for proj in self.enemy_projectiles[:]:
            if proj.distance_to(self.player_pos) < 12:
                self.enemy_projectiles.remove(proj)
                if self.shield_active:
                    # Sound: shield_hit
                    self._create_explosion(self.player_pos, 15, self.COLOR_PLAYER_SHIELD, 5)
                else:
                    # Sound: player_hit
                    self.player_health -= 15 if hasattr(proj, 'is_bomber') else 10
                    self.player_health = max(0, self.player_health)
                    self.player_hit_flash_timer = 5
                    reward -= 1.0
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_timers(self):
        if self.player_fire_cooldown_timer > 0: self.player_fire_cooldown_timer -= 1
        if self.player_hit_flash_timer > 0: self.player_hit_flash_timer -= 1
        if self.shield_timer > 0:
            self.shield_timer -= 1
            if self.shield_timer == 0:
                self.shield_active = False
                # Sound: shield_down
        if self.shield_cooldown_timer > 0: self.shield_cooldown_timer -= 1

    def _create_explosion(self, pos, num_particles, color, speed_mult=1):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4) * speed_mult
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': random.randint(10, 25),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "health": self.player_health}

    def _render_game(self):
        # Stars
        for x, y, speed in self.stars:
            new_y = (y + self.steps * speed) % self.height
            color_val = 50 + speed * 40
            pygame.draw.rect(self.screen, (color_val, color_val, color_val), (x, new_y, speed, speed))

        # Aliens
        for alien in self.aliens:
            pos = (int(alien['pos'].x), int(alien['pos'].y))
            color_map = {'basic': self.COLOR_ALIEN_BASIC, 'advanced': self.COLOR_ALIEN_ADVANCED, 'bomber': self.COLOR_ALIEN_BOMBER}
            color = color_map[alien['type']]
            if alien['type'] == 'basic':
                pygame.gfxdraw.filled_polygon(self.screen, [(pos[0]-8, pos[1]), (pos[0]+8, pos[1]), (pos[0], pos[1]-8)], color)
            elif alien['type'] == 'advanced':
                pygame.gfxdraw.filled_polygon(self.screen, [(pos[0]-10, pos[1]+5), (pos[0]+10, pos[1]+5), (pos[0], pos[1]-10)], color)
            elif alien['type'] == 'bomber':
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, color)

        # Player projectiles
        for proj in self.player_projectiles:
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE_PLAYER, (proj.x, proj.y), (proj.x, proj.y-10), 3)

        # Enemy projectiles
        for proj in self.enemy_projectiles:
            pos = (int(proj.x), int(proj.y))
            if hasattr(proj, 'is_bomber'):
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_ALIEN_BOMBER)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, self.COLOR_ALIEN_BOMBER)
            else:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_PROJECTILE_ENEMY)
        
        # Player
        if not self.game_over:
            p_pos = (int(self.player_pos.x), int(self.player_pos.y))
            color = (255, 255, 255) if self.player_hit_flash_timer > 0 else self.COLOR_PLAYER
            ship_points = [(p_pos[0], p_pos[1] - 12), (p_pos[0] - 8, p_pos[1] + 8), (p_pos[0] + 8, p_pos[1] + 8)]
            pygame.gfxdraw.aapolygon(self.screen, ship_points, color)
            pygame.gfxdraw.filled_polygon(self.screen, ship_points, color)

            # Shield
            if self.shield_active:
                alpha = 100 + int(math.sin(self.steps * 0.2) * 40)
                color = (*self.COLOR_PLAYER_SHIELD[:3], alpha)
                radius = 20 + int(math.sin(self.steps * 0.2) * 3)
                pygame.gfxdraw.filled_circle(self.screen, p_pos[0], p_pos[1], radius, color)
                pygame.gfxdraw.aacircle(self.screen, p_pos[0], p_pos[1], radius, color)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 25.0))
            color = (*p['color'], alpha)
            pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(0, int(p['life']/5)), color)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.width - score_text.get_width() - 10, 10))

        # Wave
        wave_str = f"WAVE {self.current_wave}" if self.current_wave > 0 else ""
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.width/2 - wave_text.get_width()/2, 10))
        
        # Health Bar
        health_pct = self.player_health / self.PLAYER_HEALTH_MAX
        health_bar_width = 150
        health_bar_height = 15
        pygame.draw.rect(self.screen, (50,50,50), (10, self.height - health_bar_height - 10, health_bar_width, health_bar_height))
        current_health_width = health_bar_width * health_pct
        health_color = self.COLOR_HEALTH_GREEN if health_pct > 0.3 else self.COLOR_HEALTH_RED
        if current_health_width > 0:
            pygame.draw.rect(self.screen, health_color, (10, self.height - health_bar_height - 10, current_health_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, self.height - health_bar_height - 10, health_bar_width, health_bar_height), 1)

        # Shield Cooldown Indicator
        shield_bar_width = 70
        shield_bar_height = 8
        shield_y = self.height - health_bar_height - shield_bar_height - 15
        if self.shield_cooldown_timer > 0:
            cooldown_pct = self.shield_cooldown_timer / self.SHIELD_COOLDOWN
            pygame.draw.rect(self.screen, (50,50,50), (10, shield_y, shield_bar_width, shield_bar_height))
            pygame.draw.rect(self.screen, (100,100,150), (10, shield_y, shield_bar_width * (1-cooldown_pct), shield_bar_height))
        else:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_SHIELD, (10, shield_y, shield_bar_width, shield_bar_height))
        
        # Game State Text
        if self.wave_transition_timer > 0 and self.current_wave < 3:
            text = self.font_large.render(f"WAVE {self.current_wave} CLEAR", True, self.COLOR_TEXT)
            self.screen.blit(text, (self.width/2 - text.get_width()/2, self.height/2 - text.get_height()/2))
        elif self.victory:
            text = self.font_large.render("VICTORY!", True, self.COLOR_PLAYER)
            self.screen.blit(text, (self.width/2 - text.get_width()/2, self.height/2 - text.get_height()/2))
        elif self.game_over:
            text = self.font_large.render("GAME OVER", True, self.COLOR_ALIEN_BASIC)
            self.screen.blit(text, (self.width/2 - text.get_width()/2, self.height/2 - text.get_height()/2))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Arcade Shooter")
    screen = pygame.display.set_mode((env.width, env.height))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            # Add a small delay before restarting
            pygame.time.wait(2000)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        clock.tick(30) # Match the auto_advance rate
        
    env.close()