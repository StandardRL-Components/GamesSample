import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:13:19.102040
# Source Brief: brief_00286.md
# Brief Index: 286
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a tower defense game.

    The player manages a power grid to activate defensive turrets against
    waves of alien invaders. The goal is to survive all 20 waves.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) - cycles turret selection
    - action[1]: Space button (0=released, 1=held) - toggles power on selected turret
    - action[2]: Shift button (0=released, 1=held) - currently unused

    Observation Space: Box(shape=(400, 640, 3), dtype=uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +100 for surviving all 20 waves.
    - -100 for losing all turrets.
    - +1 for each alien destroyed.
    - +0.1 for each projectile hit on an alien.
    - -0.1 for each alien collision with a turret.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your power core from alien invaders by strategically activating defensive turrets. Survive all the waves to win."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to cycle through turrets. Press space to toggle power on the selected turret."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2500
    MAX_WAVES = 20
    MAX_ACTIVE_TURRETS = 2

    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_TURRET_BASE = (60, 70, 90)
    COLOR_TURRET_BARREL = (150, 160, 180)
    COLOR_TURRET_POWERED = (0, 255, 180)
    COLOR_ALIEN = (255, 50, 100)
    COLOR_PROJECTILE = (0, 255, 255)
    COLOR_EXPLOSION = (255, 200, 50)
    COLOR_POWER_FLOW = (50, 150, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_GREEN = (40, 200, 80)
    COLOR_HEALTH_RED = (180, 40, 40)
    
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # --- Game State ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        
        self.wave_number = 0
        self.wave_transition_timer = 0

        self.turrets = []
        self.aliens = []
        self.projectiles = []
        self.particles = []
        
        self.selected_turret_idx = 0
        self.prev_space_held = False
        self.prev_movement_action = 0

        self.power_core_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20)

        # The reset call is now handled by the environment runner
        # self.reset() 
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        
        self.wave_number = 1
        self.wave_transition_timer = 120 # Time before first wave starts

        self.aliens = []
        self.projectiles = []
        self.particles = []

        self._initialize_turrets()
        self.selected_turret_idx = 0
        self.prev_space_held = False
        self.prev_movement_action = 0
        
        return self._get_observation(), self._get_info()

    def _initialize_turrets(self):
        self.turrets = []
        positions = [
            (100, self.SCREEN_HEIGHT - 80),
            (240, self.SCREEN_HEIGHT - 150),
            (400, self.SCREEN_HEIGHT - 150),
            (540, self.SCREEN_HEIGHT - 80),
        ]
        for i, pos in enumerate(positions):
            self.turrets.append({
                "id": i,
                "pos": pygame.Vector2(pos),
                "health": 100,
                "max_health": 100,
                "powered": False,
                "angle": -90,
                "range": 180,
                "fire_cooldown": 0,
                "fire_rate": 20, # frames between shots
                "damage": 10,
            })

    def step(self, action):
        reward = 0.0
        self.steps += 1
        
        if not self.game_over:
            self._handle_input(action)
            
            if self.wave_transition_timer > 0:
                self.wave_transition_timer -= 1
                if self.wave_transition_timer == 0:
                    self._spawn_wave()
            else:
                reward += self._update_turrets()
                reward += self._update_projectiles()
                reward += self._update_aliens()

            self._update_particles()
            
            if not self.aliens and self.wave_transition_timer == 0 and self.wave_number <= self.MAX_WAVES:
                if self.wave_number == self.MAX_WAVES:
                    self.win = True
                else:
                    self.wave_number += 1
                    self.wave_transition_timer = 150 # Time between waves

        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            self.game_over = True
            if self.win:
                reward += 100.0
            else: # Lost by turret destruction
                reward -= 100.0
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Turret Selection (cycle on new press) ---
        if movement != 0 and movement != self.prev_movement_action:
            if movement in [1, 4]: # Up or Right
                self.selected_turret_idx = (self.selected_turret_idx + 1) % len(self.turrets)
            elif movement in [2, 3]: # Down or Left
                self.selected_turret_idx = (self.selected_turret_idx - 1 + len(self.turrets)) % len(self.turrets)
        self.prev_movement_action = movement

        # --- Toggle Power (on press) ---
        if space_held and not self.prev_space_held:
            selected_turret = self.turrets[self.selected_turret_idx]
            if selected_turret['health'] > 0:
                if not selected_turret['powered']:
                    num_active = sum(1 for t in self.turrets if t['powered'])
                    if num_active < self.MAX_ACTIVE_TURRETS:
                        selected_turret['powered'] = True
                        # SFX: power_on.wav
                else:
                    selected_turret['powered'] = False
                    # SFX: power_off.wav
        self.prev_space_held = space_held

    def _update_turrets(self):
        for turret in self.turrets:
            if turret['health'] <= 0 or not turret['powered']:
                turret['powered'] = False
                continue

            # --- Cooldown ---
            if turret['fire_cooldown'] > 0:
                turret['fire_cooldown'] -= 1

            # --- Find Target ---
            target = None
            min_dist = float('inf')
            for alien in self.aliens:
                dist = turret['pos'].distance_to(alien['pos'])
                if dist < turret['range'] and dist < min_dist:
                    min_dist = dist
                    target = alien
            
            # --- Aim and Fire ---
            if target:
                direction = target['pos'] - turret['pos']
                turret['angle'] = math.degrees(math.atan2(-direction.y, direction.x))
                if turret['fire_cooldown'] == 0:
                    self._fire_projectile(turret)
                    turret['fire_cooldown'] = turret['fire_rate']
        return 0.0

    def _fire_projectile(self, turret):
        angle_rad = math.radians(turret['angle'])
        vel = pygame.Vector2(math.cos(angle_rad), -math.sin(angle_rad)) * 10
        start_pos = turret['pos'] + vel.normalize() * 20
        self.projectiles.append({
            "pos": start_pos,
            "vel": vel,
        })
        # SFX: laser_shoot.wav

    def _update_projectiles(self):
        reward = 0.0
        for proj in self.projectiles[:]:
            proj['pos'] += proj['vel']
            
            # --- Screen bounds check ---
            if not self.screen.get_rect().collidepoint(proj['pos']):
                self.projectiles.remove(proj)
                continue
            
            # --- Collision with aliens ---
            hit = False
            for alien in self.aliens[:]:
                if proj['pos'].distance_to(alien['pos']) < alien['size']:
                    alien['health'] -= self.turrets[0]['damage'] # Assuming all turrets have same damage
                    reward += 0.1
                    self._create_explosion(proj['pos'], 3, self.COLOR_PROJECTILE)
                    hit = True
                    # SFX: hit_impact.wav

                    if alien['health'] <= 0:
                        reward += 1.0
                        self._create_explosion(alien['pos'], 15, self.COLOR_ALIEN)
                        self.aliens.remove(alien)
                        # SFX: alien_explosion.wav
                    break 
            
            if hit:
                self.projectiles.remove(proj)
        
        return reward

    def _update_aliens(self):
        reward = 0.0
        for alien in self.aliens[:]:
            alien['pos'].y += alien['speed']
            
            # --- Collision with turrets ---
            for turret in self.turrets:
                if turret['health'] > 0 and alien['pos'].distance_to(turret['pos']) < alien['size'] + 10:
                    turret['health'] -= 25 # Damage from alien collision
                    reward -= 0.1
                    self._create_explosion(alien['pos'], 10, self.COLOR_EXPLOSION)
                    self.aliens.remove(alien)
                    # SFX: turret_damage.wav
                    break
            
            # --- Reached bottom ---
            if alien['pos'].y > self.SCREEN_HEIGHT:
                 self.aliens.remove(alien)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= 0.1
            if p['lifespan'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _spawn_wave(self):
        num_aliens = 3 + self.wave_number
        alien_health = 10 + self.wave_number * 2
        alien_speed = 0.6 + self.wave_number * 0.05
        
        for _ in range(num_aliens):
            self.aliens.append({
                "pos": pygame.Vector2(random.uniform(50, self.SCREEN_WIDTH - 50), random.uniform(-100, -20)),
                "health": alien_health,
                "max_health": alien_health,
                "speed": alien_speed,
                "size": 12
            })

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": random.uniform(2, 5),
                "lifespan": random.randint(15, 30),
                "color": color
            })

    def _check_termination(self):
        if self.win:
            return True
        if all(t['health'] <= 0 for t in self.turrets):
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_power_flow()
        self._render_particles()
        self._render_aliens()
        self._render_projectiles()
        self._render_turrets()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "turret_healths": [t['health'] for t in self.turrets],
            "active_turrets": sum(1 for t in self.turrets if t['powered']),
        }

    # --- Rendering Methods ---

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_power_flow(self):
        # Draw static grid lines from core to all turrets
        for turret in self.turrets:
            pygame.draw.line(self.screen, self.COLOR_GRID, self.power_core_pos, turret['pos'], 2)
        
        # Draw animated power flow to active turrets
        for turret in self.turrets:
            if turret['powered'] and turret['health'] > 0:
                self._draw_animated_line(self.power_core_pos, turret['pos'], self.COLOR_POWER_FLOW)

    def _draw_animated_line(self, start, end, color):
        num_dots = 10
        for i in range(num_dots):
            progress = ((self.steps * 2 + i * 5) % 50) / 50.0
            pos = start.lerp(end, progress)
            self._draw_glow_circle(self.screen, color, pos, 3, 0.5)

    def _render_turrets(self):
        for i, turret in enumerate(self.turrets):
            if turret['health'] <= 0:
                # Render destroyed turret
                self._draw_glow_circle(self.screen, (50,50,50), turret['pos'], 18, 0.3)
                pygame.gfxdraw.filled_circle(self.screen, int(turret['pos'].x), int(turret['pos'].y), 12, (30,30,30))
                continue

            # Selection indicator
            if i == self.selected_turret_idx:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                radius = 25 + pulse * 5
                alpha = int(70 + pulse * 30)
                color = self.COLOR_TURRET_POWERED if turret['powered'] else (255, 255, 255)
                pygame.gfxdraw.aacircle(self.screen, int(turret['pos'].x), int(turret['pos'].y), int(radius), (*color, alpha))
            
            # Range indicator for powered turrets
            if turret['powered']:
                pygame.gfxdraw.aacircle(self.screen, int(turret['pos'].x), int(turret['pos'].y), int(turret['range']), (*self.COLOR_TURRET_POWERED, 40))

            # Base
            color = self.COLOR_TURRET_POWERED if turret['powered'] else self.COLOR_TURRET_BASE
            self._draw_glow_circle(self.screen, color, turret['pos'], 18, 0.5)
            pygame.gfxdraw.filled_circle(self.screen, int(turret['pos'].x), int(turret['pos'].y), 12, self.COLOR_TURRET_BASE)
            pygame.gfxdraw.aacircle(self.screen, int(turret['pos'].x), int(turret['pos'].y), 12, color)

            # Barrel
            angle_rad = math.radians(turret['angle'])
            end_pos = turret['pos'] + pygame.Vector2(math.cos(angle_rad), -math.sin(angle_rad)) * 20
            pygame.draw.line(self.screen, self.COLOR_TURRET_BARREL, turret['pos'], end_pos, 5)

            # Health bar
            bar_width = 30
            bar_height = 5
            bar_pos = (turret['pos'].x - bar_width / 2, turret['pos'].y - 30)
            health_ratio = max(0, turret['health'] / turret['max_health'])
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (*bar_pos, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (*bar_pos, bar_width * health_ratio, bar_height))

    def _render_aliens(self):
        for alien in self.aliens:
            self._draw_glow_circle(self.screen, self.COLOR_ALIEN, alien['pos'], alien['size'] + 4, 0.7)
            pygame.gfxdraw.filled_circle(self.screen, int(alien['pos'].x), int(alien['pos'].y), alien['size'], self.COLOR_ALIEN)
            pygame.gfxdraw.aacircle(self.screen, int(alien['pos'].x), int(alien['pos'].y), alien['size'], (255,150,180))

    def _render_projectiles(self):
        for proj in self.projectiles:
            self._draw_glow_circle(self.screen, self.COLOR_PROJECTILE, proj['pos'], 6, 1.0)
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, proj['pos'], proj['pos'] - proj['vel'].normalize()*8, 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = p['lifespan'] / 30.0
            color = (*p['color'], int(alpha * 255))
            self._draw_glow_circle(self.screen, p['color'], p['pos'], p['radius'] + 2, alpha * 0.5)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Power status
        power_text = self.font_small.render(f"POWER: {sum(1 for t in self.turrets if t['powered'])}/{self.MAX_ACTIVE_TURRETS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(power_text, (self.SCREEN_WIDTH / 2 - power_text.get_width()/2, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "MISSION COMPLETE" if self.win else "GAME OVER"
            color = self.COLOR_TURRET_POWERED if self.win else self.COLOR_ALIEN
            end_text = self.font_large.render(msg, True, color)
            pos = (self.SCREEN_WIDTH / 2 - end_text.get_width() / 2, self.SCREEN_HEIGHT / 2 - end_text.get_height() / 2)
            self.screen.blit(end_text, pos)
        elif self.wave_transition_timer > 0:
            msg = f"WAVE {self.wave_number} INCOMING"
            alpha = min(255, int(255 * (self.wave_transition_timer / 60.0))) if self.wave_transition_timer < 60 else 255
            wave_announce_text = self.font_large.render(msg, True, (*self.COLOR_UI_TEXT, alpha))
            pos = (self.SCREEN_WIDTH / 2 - wave_announce_text.get_width() / 2, self.SCREEN_HEIGHT / 2 - wave_announce_text.get_height() / 2)
            wave_announce_text.set_alpha(alpha)
            self.screen.blit(wave_announce_text, pos)


    def _draw_glow_circle(self, surface, color, center, radius, intensity):
        if intensity <= 0: return
        num_layers = 5
        for i in range(num_layers, 0, -1):
            layer_radius = int(radius + (i * 2) * intensity)
            alpha = int(40 * (1 - i / num_layers) * intensity)
            if layer_radius > 0:
                pygame.gfxdraw.aacircle(surface, int(center.x), int(center.y), layer_radius, (*color, alpha))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # You need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # Arrow keys: Cycle turret selection
    # Space: Toggle power
    # Q: Quit
    
    action = [0, 0, 0] # [movement, space, shift]
    
    pygame.display.set_caption("Tower Defense Gym Environment")
    display_surface = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                # Map keyboard to MultiDiscrete action components
                if event.key == pygame.K_UP: action[0] = 1
                if event.key == pygame.K_DOWN: action[0] = 2
                if event.key == pygame.K_LEFT: action[0] = 3
                if event.key == pygame.K_RIGHT: action[0] = 4
                if event.key == pygame.K_SPACE: action[1] = 1
                if event.key == pygame.K_LSHIFT: action[2] = 1
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    action[0] = 0
                if event.key == pygame.K_SPACE: action[1] = 0
                if event.key == pygame.K_LSHIFT: action[2] = 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render for human viewing ---
        # The observation is already the rendered screen, but transposed.
        # So we transpose it back for pygame's display format.
        frame = np.transpose(obs, (1, 0, 2))
        
        pygame.surfarray.blit_array(display_surface, frame)
        pygame.display.flip()
        env.clock.tick(30) # Limit to 30 FPS

    env.close()