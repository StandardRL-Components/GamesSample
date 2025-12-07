import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:05:48.719692
# Source Brief: brief_00256.md
# Brief Index: 256
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, color, lifetime=20):
        self.pos = list(pos)
        self.vel = [random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)]
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            radius = int(3 * (self.lifetime / self.max_lifetime))
            if radius > 0:
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, self.color + (alpha,), (radius, radius), radius)
                surface.blit(temp_surf, (int(self.pos[0]) - radius, int(self.pos[1]) - radius), special_flags=pygame.BLEND_RGBA_ADD)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Move elemental orbs into colored zones and activate them to generate power. "
        "Discover recipes and synchronize activations to win before time expires."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the selector between orbs and zones. "
        "Press space to pick up or drop an orb. Press shift to activate a zone."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    TARGET_POWER = 100

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BAR_BG = (40, 50, 70)
    COLOR_UI_BAR_FILL = (100, 200, 255)
    COLOR_SELECTOR = (0, 255, 255)

    ORB_COLORS = {
        "fire": (255, 80, 0),
        "water": (0, 150, 255),
        "earth": (80, 200, 50),
        "air": (255, 255, 100),
        "combined": (255, 255, 255)
    }
    ZONE_COLORS = {
        "red": (120, 0, 0),
        "green": (0, 120, 0),
        "blue": (0, 0, 120),
        "yellow": (120, 120, 0)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_orb = pygame.font.SysFont("Arial", 14)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.orbs = []
        self.zones = []
        self.particles = []
        self.selector_idx = 0
        self.held_orb_id = None
        self.selectable_items = []
        
        # Action press state trackers
        self.space_was_held = False
        self.shift_was_held = False
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.held_orb_id = None
        self.particles = []
        self.space_was_held = True # Prevent action on first frame
        self.shift_was_held = True

        # Initialize Zones
        zone_size = 120
        margin = 40
        zone_y_top = 80
        zone_y_bottom = self.SCREEN_HEIGHT - margin - zone_size
        zone_x_left = self.SCREEN_WIDTH // 2 - margin // 2 - zone_size
        zone_x_right = self.SCREEN_WIDTH // 2 + margin // 2

        self.zones = [
            {'id': 0, 'type': 'red', 'rect': pygame.Rect(zone_x_left, zone_y_top, zone_size, zone_size), 'color': self.ZONE_COLORS['red'], 'activation_timer': 0},
            {'id': 1, 'type': 'green', 'rect': pygame.Rect(zone_x_right, zone_y_top, zone_size, zone_size), 'color': self.ZONE_COLORS['green'], 'activation_timer': 0},
            {'id': 2, 'type': 'blue', 'rect': pygame.Rect(zone_x_left, zone_y_bottom, zone_size, zone_size), 'color': self.ZONE_COLORS['blue'], 'activation_timer': 0},
            {'id': 3, 'type': 'yellow', 'rect': pygame.Rect(zone_x_right, zone_y_bottom, zone_size, zone_size), 'color': self.ZONE_COLORS['yellow'], 'activation_timer': 0},
        ]

        # Initialize Orbs
        orb_types = ["fire", "water", "earth", "air"]
        self.orbs = []
        start_x = 80
        for i, orb_type in enumerate(orb_types):
            self.orbs.append({
                'id': i,
                'type': orb_type,
                'color': self.ORB_COLORS[orb_type],
                'pos': [start_x + i * 50, self.SCREEN_HEIGHT - 40],
                'target_pos': [start_x + i * 50, self.SCREEN_HEIGHT - 40],
                'radius': 15,
                'zone_id': None,
                'is_combo_orb': False,
            })
        
        self._update_selectable_items()
        self.selector_idx = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # Unpack actions
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.space_was_held
        shift_pressed = shift_held and not self.shift_was_held
        
        # --- Handle Input and Game Logic ---
        
        # 1. Update Selector
        if movement != 0:
            self._update_selector(movement)

        # 2. Handle Primary Action (Space) - Pick up/Drop Orb
        if space_pressed:
            reward += self._handle_primary_action()

        # 3. Handle Secondary Action (Shift) - Activate Zone
        activated_zones_this_step = []
        if shift_pressed:
            activated_zones_this_step, combo_reward, combo_power = self._handle_secondary_action()
            reward += combo_reward
            self.score += combo_power
        
        # 4. Synchronization Bonus
        if len(activated_zones_this_step) >= 2:
            # // SFX: Sync bonus
            reward += 0.5
            self.score += 10
        
        # --- Update Game State ---
        self._update_orbs()
        self._update_zones()
        self._update_particles()
        
        # --- Calculate Reward and Termination ---
        self.score = min(self.TARGET_POWER, self.score)
        
        terminated = False
        if self.score >= self.TARGET_POWER:
            # // SFX: Victory
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            # // SFX: Failure
            reward -= 10.0
            terminated = True
            self.game_over = True

        # Store action state for next step
        self.space_was_held = space_held
        self.shift_was_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_selector(self, movement):
        if not self.selectable_items:
            return

        current_pos = self.selectable_items[self.selector_idx]['pos']
        best_target_idx = self.selector_idx
        min_dist = float('inf')

        # Directions: 1=up, 2=down, 3=left, 4=right
        for i, item in enumerate(self.selectable_items):
            if i == self.selector_idx:
                continue
            
            dx = item['pos'][0] - current_pos[0]
            dy = item['pos'][1] - current_pos[1]

            is_in_direction = False
            if movement == 1 and dy < 0 and abs(dy) > abs(dx): is_in_direction = True # Up
            elif movement == 2 and dy > 0 and abs(dy) > abs(dx): is_in_direction = True # Down
            elif movement == 3 and dx < 0 and abs(dx) > abs(dy): is_in_direction = True # Left
            elif movement == 4 and dx > 0 and abs(dx) > abs(dy): is_in_direction = True # Right

            if is_in_direction:
                dist = math.hypot(dx, dy)
                if dist < min_dist:
                    min_dist = dist
                    best_target_idx = i
        
        self.selector_idx = best_target_idx

    def _handle_primary_action(self):
        if not self.selectable_items or self.selector_idx < 0:
            return 0.0
        selected_item = self.selectable_items[self.selector_idx]

        if self.held_orb_id is None:
            # Try to pick up an orb
            if selected_item['type'] == 'orb':
                # // SFX: Orb pickup
                self.held_orb_id = selected_item['id']
                self._update_selectable_items()
                # Find the new index of the previously selected zone, or default to 0
                try:
                    self.selector_idx = [item['id'] for item in self.selectable_items if item['type'] == 'zone'].index(selected_item['last_zone_id'])
                except (ValueError, KeyError):
                    self.selector_idx = 0
                return 0.0
        else:
            # Try to drop an orb in a zone
            if selected_item['type'] == 'zone':
                # // SFX: Orb drop
                orb = next(o for o in self.orbs if o['id'] == self.held_orb_id)
                orb['zone_id'] = selected_item['id']
                
                zone_rect = self.zones[orb['zone_id']]['rect']
                buffer = orb['radius'] + 5
                target_x = self.np_random.integers(zone_rect.left + buffer, zone_rect.right - buffer + 1)
                target_y = self.np_random.integers(zone_rect.top + buffer, zone_rect.bottom - buffer + 1)
                orb['target_pos'] = [target_x, target_y]

                self.held_orb_id = None
                self._update_selectable_items()
                return 0.0
        return 0.0

    def _handle_secondary_action(self):
        if not self.selectable_items or self.selector_idx < 0:
            return [], 0.0, 0
        selected_item = self.selectable_items[self.selector_idx]
        activated_zones = []
        reward = 0.0
        power = 0

        if selected_item['type'] == 'zone':
            zone_id = selected_item['id']
            zone = self.zones[zone_id]
            
            if zone['activation_timer'] <= 0:
                # // SFX: Zone activate
                zone['activation_timer'] = 30 # Pulse for 1 second at 30fps
                activated_zones.append(zone_id)
                
                orbs_in_zone = [o for o in self.orbs if o['zone_id'] == zone_id and not o['is_combo_orb']]
                
                if len(orbs_in_zone) >= 2:
                    orb_types_in_zone = {o['type'] for o in orbs_in_zone}
                    
                    combo_found = False
                    # Red+Green in Blue Zone -> Power +20 -> fire + water in blue
                    if zone['type'] == 'blue' and {'fire', 'water'}.issubset(orb_types_in_zone):
                        power, combo_found = 20, True
                        used_types = {'fire', 'water'}
                    # Red+Blue in Green Zone -> Power +30 -> fire + earth in green
                    elif zone['type'] == 'green' and {'fire', 'earth'}.issubset(orb_types_in_zone):
                        power, combo_found = 30, True
                        used_types = {'fire', 'earth'}
                    # Green+Blue in Red Zone -> Power +40 -> water + earth in red
                    elif zone['type'] == 'red' and {'water', 'earth'}.issubset(orb_types_in_zone):
                        power, combo_found = 40, True
                        used_types = {'water', 'earth'}

                    if combo_found:
                        # // SFX: Combination success
                        reward += 1.0 + (power * 0.1)
                        orbs_to_remove = []
                        for orb_type in used_types:
                            orb_to_remove = next((o for o in orbs_in_zone if o['type'] == orb_type and o['id'] not in [r['id'] for r in orbs_to_remove]), None)
                            if orb_to_remove:
                                orbs_to_remove.append(orb_to_remove)
                        
                        used_orb_ids = [o['id'] for o in orbs_to_remove]
                        self.orbs = [o for o in self.orbs if o['id'] not in used_orb_ids]
                        
                        new_id = max([o['id'] for o in self.orbs] or [-1]) + 1
                        self.orbs.append({
                            'id': new_id, 'type': 'combined', 'color': self.ORB_COLORS['combined'],
                            'pos': list(zone['rect'].center), 'target_pos': list(zone['rect'].center),
                            'radius': 20, 'zone_id': zone_id, 'is_combo_orb': True
                        })
                        self._update_selectable_items()
                    else:
                        # // SFX: Combination fail
                        pass
        return activated_zones, reward, power

    def _update_orbs(self):
        for orb in self.orbs:
            dx = orb['target_pos'][0] - orb['pos'][0]
            dy = orb['target_pos'][1] - orb['pos'][1]
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                orb['pos'][0] += dx * 0.1
                orb['pos'][1] += dy * 0.1
            else:
                orb['pos'] = list(orb['target_pos'])

            if self.np_random.random() < 0.5:
                self.particles.append(Particle(orb['pos'], orb['color']))
    
    def _update_zones(self):
        for zone in self.zones:
            if zone['activation_timer'] > 0:
                zone['activation_timer'] -= 1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

    def _update_selectable_items(self):
        self.selectable_items = []
        for z in self.zones:
            self.selectable_items.append({'type': 'zone', 'id': z['id'], 'pos': z['rect'].center})
        
        last_selected_zone_id = self.selector_idx if self.selector_idx < len(self.zones) else 0
        for o in self.orbs:
            if o['zone_id'] is None and o['id'] != self.held_orb_id:
                self.selectable_items.append({'type': 'orb', 'id': o['id'], 'pos': o['pos'], 'last_zone_id': last_selected_zone_id})
        
        if self.selector_idx >= len(self.selectable_items):
            self.selector_idx = 0 if self.selectable_items else -1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        self._render_zones()
        self._render_particles()
        self._render_orbs()
        self._render_selector()

    def _render_zones(self):
        for zone in self.zones:
            pygame.draw.rect(self.screen, zone['color'], zone['rect'], 2, border_radius=5)
            if zone['activation_timer'] > 0:
                progress = zone['activation_timer'] / 30.0
                alpha = int(150 * math.sin(progress * math.pi))
                glow_color = zone['color'] + (alpha,)
                
                glow_surf = pygame.Surface(zone['rect'].size, pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), 4, border_radius=5)
                self.screen.blit(glow_surf, zone['rect'].topleft)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_orbs(self):
        for orb in self.orbs:
            pos = [int(p) for p in orb['pos']]
            radius = orb['radius']
            
            glow_radius = int(radius * 1.8)
            glow_alpha = 80
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, orb['color'] + (glow_alpha,), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, orb['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, orb['color'])
            
            if not orb['is_combo_orb']:
                initial = orb['type'][0].upper()
                text_surf = self.font_orb.render(initial, True, self.COLOR_BG)
                text_rect = text_surf.get_rect(center=pos)
                self.screen.blit(text_surf, text_rect)

    def _render_selector(self):
        if not self.selectable_items or self.selector_idx < 0: return
        
        item = self.selectable_items[self.selector_idx]
        
        if item['type'] == 'zone':
            rect = self.zones[item['id']]['rect'].inflate(10, 10)
            pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, 2, border_radius=8)
        elif item['type'] == 'orb':
            orb = next((o for o in self.orbs if o['id'] == item['id']), None)
            if orb:
                pos = [int(p) for p in orb['pos']]
                radius = orb['radius'] + 6
                pygame.draw.circle(self.screen, self.COLOR_SELECTOR, pos, radius, 2)
        
        if self.held_orb_id is not None:
            orb = next((o for o in self.orbs if o['id'] == self.held_orb_id), None)
            if orb:
                pos = [self.SCREEN_WIDTH - 40, 40]
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], orb['radius'], orb['color'])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], orb['radius'], orb['color'])
                pygame.draw.circle(self.screen, self.COLOR_UI_TEXT, pos, orb['radius']+2, 1)

    def _render_ui(self):
        bar_width, bar_height = 400, 20
        bar_x, bar_y = (self.SCREEN_WIDTH - bar_width) // 2, 20
        fill_ratio = self.score / self.TARGET_POWER
        fill_width = int(bar_width * fill_ratio)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (bar_x, bar_y, fill_width, bar_height), border_radius=5)
        
        power_text = self.font_ui.render(f"POWER: {int(self.score)} / {self.TARGET_POWER}", True, self.COLOR_UI_TEXT)
        self.screen.blit(power_text, power_text.get_rect(center=(self.SCREEN_WIDTH // 2, bar_y + bar_height // 2)))

        remaining_seconds = (self.MAX_STEPS - self.steps) / 30.0
        timer_text = self.font_ui.render(f"TIME: {max(0, remaining_seconds):.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10)))

    def close(self):
        pygame.font.quit()
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    # This check is necessary for the __main__ block to work, but not for the environment itself
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Orb Synchronizer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    last_movement_time = 0
    MOVEMENT_DELAY = 150 # ms

    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        current_time = pygame.time.get_ticks()
        
        if current_time - last_movement_time > MOVEMENT_DELAY:
            moved = False
            if keys[pygame.K_UP]: 
                movement = 1
                moved = True
            elif keys[pygame.K_DOWN]: 
                movement = 2
                moved = True
            elif keys[pygame.K_LEFT]: 
                movement = 3
                moved = True
            elif keys[pygame.K_RIGHT]: 
                movement = 4
                moved = True
            if moved:
                last_movement_time = current_time

        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # The environment returns an RGB array, we need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            env.reset()
            pygame.time.wait(2000) # Pause before restarting
            
        clock.tick(30) # Run at 30 FPS

    env.close()