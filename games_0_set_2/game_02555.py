import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A tower defense game where the player places towers to defend a base from waves of aliens.
    The game is implemented as a Gymnasium environment with a MultiDiscrete action space.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place a tower. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of descending aliens by strategically placing defensive towers."
    )

    # Frames auto-advance at 30fps for smooth, real-time gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 30 * 120 # 2 minutes max
        self.MAX_WAVES = 10

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
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Colors for high contrast and visual clarity
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_BUILD_ZONE = (30, 30, 50)
        self.COLOR_SPAWN_ZONE = (50, 20, 20)
        self.COLOR_BASE = (50, 150, 50)
        self.COLOR_ALIEN = (220, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TOWER = [(80, 120, 255), (255, 120, 80)] # Standard, Sniper
        self.COLOR_INVALID = (255, 0, 0, 100)

        # Game constants
        self.BASE_HEALTH_MAX = 10
        self.CURSOR_SPEED = 8
        self.WAVE_COOLDOWN = 3 * self.FPS # 3 seconds between waves

        # Tower specifications
        self.TOWER_SPECS = [
            {"cost": 100, "range": 80, "damage": 1, "fire_rate": 15, "color": self.COLOR_TOWER[0]}, # Standard
            {"cost": 250, "range": 150, "damage": 5, "fire_rate": 60, "color": self.COLOR_TOWER[1]}, # Sniper
        ]

        # Define build zones
        self.build_zones = [
            pygame.Rect(50, 80, 150, self.HEIGHT - 120),
            pygame.Rect(self.WIDTH - 200, 80, 150, self.HEIGHT - 120)
        ]
        
        # Initialize all state variables to prevent uninitialized attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.wave_timer = 0
        self.aliens = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.shift_was_held = False
        self.np_random = None

        self.reset()
        # self.validate_implementation() # Commented out to avoid print on load

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = self.BASE_HEALTH_MAX
        self.resources = 250
        self.wave_number = 0 # Will be incremented to 1 on first wave spawn
        self.wave_timer = self.WAVE_COOLDOWN

        self.aliens = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.selected_tower_type = 0
        self.shift_was_held = False

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01 # Time penalty per step
        
        self._handle_input(movement, space_held, shift_held)
        
        if not self.game_over:
            reward += self._update_game_state()
            self.score += reward

        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if self.game_won:
                reward += 100 # Goal-oriented reward for winning
                self.score += 100
            else: # Lost by base destruction or timeout
                reward -= 100 # Goal-oriented penalty for losing
                self.score -= 100
            self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        """Processes player actions from the action space."""
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # Cycle tower type on shift PRESS (rising edge detection)
        if shift_held and not self.shift_was_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        self.shift_was_held = shift_held
        
        # Place tower on space held
        if space_held:
            self._place_tower()

    def _place_tower(self):
        """Handles the logic for placing a new tower."""
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.resources < spec["cost"]:
            return

        is_in_zone = any(zone.collidepoint(self.cursor_pos) for zone in self.build_zones)
        if not is_in_zone:
            return

        is_overlapping = any(math.dist(self.cursor_pos, t['pos']) < 20 for t in self.towers)
        if is_overlapping:
            return
            
        self.resources -= spec["cost"]
        self.towers.append({
            "pos": list(self.cursor_pos),
            "type": self.selected_tower_type,
            "cooldown": 0,
        })
        # Add a visual effect for placement
        self.particles.append({"pos": list(self.cursor_pos), "radius": 10, "max_radius": 30, "life": 15, "color": spec['color']})
        # sfx: place_tower.wav

    def _update_game_state(self):
        """Advances the game simulation by one frame."""
        step_reward = 0
        
        # Wave management
        if not self.aliens and self.wave_number < self.MAX_WAVES:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.wave_number += 1
                self._spawn_wave()
                self.wave_timer = self.WAVE_COOLDOWN

        # Update towers
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                spec = self.TOWER_SPECS[tower['type']]
                target = None
                min_dist = spec['range']
                for alien in self.aliens:
                    d = math.dist(tower['pos'], alien['pos'])
                    if d < min_dist:
                        min_dist = d
                        target = alien
                
                if target:
                    # Fire projectile
                    tower['cooldown'] = spec['fire_rate']
                    angle = math.atan2(target['pos'][1] - tower['pos'][1], target['pos'][0] - tower['pos'][0])
                    self.projectiles.append({
                        "pos": list(tower['pos']),
                        "vel": [math.cos(angle) * 10, math.sin(angle) * 10],
                        "damage": spec['damage']
                    })
                    # sfx: laser_fire.wav

        # Update projectiles and check for collisions
        for proj in self.projectiles[:]:
            proj['pos'][0] += proj['vel'][0]
            proj['pos'][1] += proj['vel'][1]

            if not (0 < proj['pos'][0] < self.WIDTH and 0 < proj['pos'][1] < self.HEIGHT):
                self.projectiles.remove(proj)
                continue

            proj_rect = pygame.Rect(proj['pos'][0]-2, proj['pos'][1]-2, 4, 4)
            for alien in self.aliens[:]:
                alien_rect = pygame.Rect(alien['pos'][0]-8, alien['pos'][1]-8, 16, 16)
                if alien_rect.colliderect(proj_rect):
                    alien['health'] -= proj['damage']
                    step_reward += 0.1 # Reward for hitting an alien
                    self.particles.append({"pos": list(proj['pos']), "radius": 0, "max_radius": 10, "life": 5, "color": (255, 255, 255)})
                    # sfx: hit_confirm.wav
                    if proj in self.projectiles: self.projectiles.remove(proj)

                    if alien['health'] <= 0:
                        step_reward += 1.0 # Reward for destroying an alien
                        self.resources += 20 # Resource gain on kill
                        self.particles.append({"pos": list(alien['pos']), "radius": 5, "max_radius": 25, "life": 10, "color": self.COLOR_ALIEN})
                        # sfx: explosion.wav
                        self.aliens.remove(alien)
                    break

        # Update aliens and check for reaching the base
        base_rect = pygame.Rect(self.WIDTH/2 - 50, self.HEIGHT - 40, 100, 20)
        for alien in self.aliens[:]:
            alien['pos'][1] += alien['speed']
            alien_rect = pygame.Rect(alien['pos'][0]-8, alien['pos'][1]-8, 16, 16)
            if base_rect.colliderect(alien_rect):
                self.base_health -= 1
                step_reward -= 10 # Penalty for base taking damage
                self.aliens.remove(alien)
                self.particles.append({"pos": [self.WIDTH/2, self.HEIGHT-30], "radius": 10, "max_radius": 50, "life": 20, "color": (255, 80, 80)})
                # sfx: base_hit.wav

        # Update visual effect particles
        for p in self.particles[:]:
            p['life'] -= 1
            p['radius'] += (p['max_radius'] - p['radius']) * 0.2
            if p['life'] <= 0:
                self.particles.remove(p)

        return step_reward

    def _spawn_wave(self):
        """Creates a new wave of aliens with increasing difficulty."""
        num_aliens = 10 + (self.wave_number - 1) * 2
        alien_speed = 0.5 + self.wave_number * 0.1
        for _ in range(num_aliens):
            self.aliens.append({
                "pos": [random.uniform(50, self.WIDTH - 50), random.uniform(-100, -20)],
                "health": 2 + self.wave_number,
                "speed": alien_speed * random.uniform(0.8, 1.2)
            })

    def _check_termination(self):
        """Checks for game over conditions."""
        if self.base_health <= 0:
            return True
        if self.wave_number >= self.MAX_WAVES and not self.aliens:
            self.game_won = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        """Renders the current game state to a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        # Convert from (width, height, channels) to (height, width, channels)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders all visual elements of the game world."""
        # Draw static zones
        pygame.draw.rect(self.screen, self.COLOR_SPAWN_ZONE, (0, 0, self.WIDTH, 40))
        for zone in self.build_zones:
            pygame.draw.rect(self.screen, self.COLOR_BUILD_ZONE, zone)

        # Draw base
        base_rect = pygame.Rect(self.WIDTH/2 - 50, self.HEIGHT - 40, 100, 20)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.gfxdraw.rectangle(self.screen, base_rect, (*self.COLOR_BASE, 150))

        # Draw towers and their ranges (if cursor is nearby)
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            pos_int = (int(tower['pos'][0]), int(tower['pos'][1]))
            if math.dist(self.cursor_pos, tower['pos']) < spec['range']:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], spec['range'], (*spec['color'][:3], 20))
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], spec['range'], (*spec['color'][:3], 80))
            
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 10, spec['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 10, (255,255,255))

        # Draw aliens
        for alien in self.aliens:
            pos_int = (int(alien['pos'][0]), int(alien['pos'][1]))
            pts = [(pos_int[0], pos_int[1] - 8), (pos_int[0] - 8, pos_int[1] + 8), (pos_int[0] + 8, pos_int[1] + 8)]
            pygame.gfxdraw.aapolygon(self.screen, pts, self.COLOR_ALIEN)
            pygame.gfxdraw.filled_polygon(self.screen, pts, self.COLOR_ALIEN)

        # Draw projectiles
        for proj in self.projectiles:
            pos_int = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, (pos_int[0]-2, pos_int[1]-2, 4, 4))
        
        # Draw particles for visual effects
        for p in self.particles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / p['max_radius']))
            color = (*p['color'], max(0, min(255, alpha)))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(p['radius']), color)

        # Draw placement cursor with placement validity feedback
        spec = self.TOWER_SPECS[self.selected_tower_type]
        cursor_pos_int = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        
        can_place = (self.resources >= spec['cost'] and 
                     any(zone.collidepoint(self.cursor_pos) for zone in self.build_zones) and
                     not any(math.dist(self.cursor_pos, t['pos']) < 20 for t in self.towers))

        color = self.COLOR_CURSOR if can_place else self.COLOR_INVALID
        
        pygame.gfxdraw.filled_circle(self.screen, cursor_pos_int[0], cursor_pos_int[1], spec['range'], (*color[:3], 20))
        pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], spec['range'], (*color[:3], 80))
        pygame.gfxdraw.filled_circle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 10, color)
        pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 10, (255,255,255))

    def _render_ui(self):
        """Renders the user interface overlay."""
        def draw_text(text, pos, font, color, align="left"):
            surface = font.render(text, True, color)
            rect = surface.get_rect()
            if align == "left": rect.topleft = pos
            elif align == "right": rect.topright = pos
            elif align == "center": rect.center = pos
            self.screen.blit(surface, rect)

        draw_text(f"Wave: {self.wave_number}/{self.MAX_WAVES}", (10, 10), self.font_small, self.COLOR_TEXT)
        draw_text(f"Score: {int(self.score)}", (10, 30), self.font_small, self.COLOR_TEXT)
        draw_text(f"Steps: {self.steps}/{self.MAX_STEPS}", (self.WIDTH - 10, 10), self.font_small, self.COLOR_TEXT, align="right")
        draw_text(f"Resources: ${self.resources}", (10, self.HEIGHT - 30), self.font_small, self.COLOR_TEXT)
        
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_name = "Standard" if self.selected_tower_type == 0 else "Sniper"
        draw_text(f"Selected: {tower_name} (${spec['cost']})", (self.WIDTH/2, self.HEIGHT-30), self.font_small, self.COLOR_TEXT, align="center")

        # Base health bar
        health_ratio = self.base_health / self.BASE_HEALTH_MAX
        health_color = (255 * (1 - health_ratio), 255 * health_ratio, 0)
        bar_width = 150
        bar_rect = pygame.Rect(self.WIDTH - 10 - bar_width, 30, max(0, bar_width * health_ratio), 15)
        pygame.draw.rect(self.screen, health_color, bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.WIDTH - 10 - bar_width, 30, bar_width, 15), 1)
        draw_text("Base Health", (self.WIDTH - 15 - bar_width, 30), self.font_small, self.COLOR_TEXT, align="right")
        
        # Game over / Win screen overlay
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            draw_text(message, (self.WIDTH/2, self.HEIGHT/2 - 20), self.font_large, (255, 255, 100), align="center")
            draw_text(f"Final Score: {int(self.score)}", (self.WIDTH/2, self.HEIGHT/2 + 30), self.font_small, self.COLOR_TEXT, align="center")

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    # This block allows you to play the game directly for testing and visualization.
    # To run, you will need to unset the dummy video driver. For example:
    # del os.environ["SDL_VIDEODRIVER"]
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Mapping from Pygame keys to MultiDiscrete action components
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        # Handle movement input
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one movement direction per frame
        
        # Handle button inputs
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Convert observation back to a Pygame surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause to show final screen
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()