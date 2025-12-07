import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:13:45.769398
# Source Brief: brief_01604.md
# Brief Index: 1604
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a tower defense game.
    The player places fungal spore portals to defend a mushroom fortress
    against waves of hungry slugs.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your mushroom fortress from waves of hungry slugs by strategically placing fungal spore portals. "
        "Unlock different spore types to slow, heal, and explode your way to victory."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor, space to place a portal, and shift to cycle portal types. "
        "Release all keys to start the wave."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2500
    MAX_WAVES = 20
    FORTRESS_MAX_HEALTH = 100
    
    # Colors
    COLOR_BG = (15, 25, 40)
    COLOR_FORTRESS = (139, 69, 19)
    COLOR_FORTRESS_CAP = (210, 105, 30)
    COLOR_FORTRESS_HEALTH = (46, 204, 113)
    COLOR_FORTRESS_DAMAGE = (231, 76, 60)
    COLOR_SLUG = (140, 220, 100)
    COLOR_SLUG_HIT = (255, 255, 255)
    COLOR_PATH = (30, 50, 70)
    COLOR_PORTAL_SPOT = (50, 70, 100)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_UI_TEXT = (220, 220, 220)
    
    SPORE_TYPES = {
        'explosive': {'color': (255, 80, 80), 'unlock_wave': 1, 'name': 'EXPLOSIVE'},
        'healing': {'color': (80, 150, 255), 'unlock_wave': 3, 'name': 'HEALING'},
        'slowing': {'color': (150, 80, 255), 'unlock_wave': 6, 'name': 'SLOWING'},
    }
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fortress_pos = np.array([80, self.SCREEN_HEIGHT / 2], dtype=float)
        self.fortress_health = 0
        self.wave_number = 0
        self.slugs = []
        self.portals = []
        self.particles = []
        self.path = []
        self.portal_spots = []
        self.game_phase = "PLACEMENT" # "PLACEMENT" or "WAVE_ACTIVE"
        self.cursor_spot_index = 0
        self.selected_portal_type_index = 0
        self.available_spore_types = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.wave_timer = 0
        self.wave_complete_reward = 0
        self.fortress_hit_flash = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fortress_health = self.FORTRESS_MAX_HEALTH
        self.wave_number = 1
        self.slugs = []
        self.portals = []
        self.particles = []
        self.game_phase = "PLACEMENT"
        self.cursor_spot_index = 0
        self.selected_portal_type_index = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.wave_timer = 0
        self.fortress_hit_flash = 0
        
        self._generate_path()
        self._generate_portal_spots()
        self._update_available_spores()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        self.fortress_hit_flash = max(0, self.fortress_hit_flash - 1)

        # --- Action Processing ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        if self.game_phase == "PLACEMENT":
            # Move cursor
            if movement == 1: self.cursor_spot_index = (self.cursor_spot_index - 1) % len(self.portal_spots)
            if movement == 2: self.cursor_spot_index = (self.cursor_spot_index + 1) % len(self.portal_spots)
            # For left/right, we can just use the same logic for simplicity
            if movement == 3: self.cursor_spot_index = (self.cursor_spot_index - 1) % len(self.portal_spots)
            if movement == 4: self.cursor_spot_index = (self.cursor_spot_index + 1) % len(self.portal_spots)

            # Cycle spore type
            if shift_press:
                self.selected_portal_type_index = (self.selected_portal_type_index + 1) % len(self.available_spore_types)
                # sfx: UI_cycle_sound

            # Place portal
            if space_press:
                cursor_pos = self.portal_spots[self.cursor_spot_index]
                is_occupied = any(np.array_equal(p['pos'], cursor_pos) for p in self.portals)
                if not is_occupied:
                    spore_key = list(self.SPORE_TYPES.keys())[self.available_spore_types[self.selected_portal_type_index]]
                    self.portals.append({
                        'pos': np.array(cursor_pos, dtype=float),
                        'type': spore_key,
                        'cooldown': 0,
                        'animation_angle': self.np_random.uniform(0, 2 * math.pi)
                    })
                    # sfx: portal_place_sound
            
            # Start wave
            if movement == 0 and not space_held and not shift_held:
                self.game_phase = "WAVE_ACTIVE"
                self._spawn_wave()
                # sfx: wave_start_horn

        elif self.game_phase == "WAVE_ACTIVE":
            # --- Game Logic Updates ---
            reward += self._update_portals()
            reward += self._update_slugs()
            self._update_particles()
            
            # --- Wave Completion Check ---
            if not self.slugs and self.wave_timer <= 0:
                reward += 5 # Wave defeated reward
                self.wave_number += 1
                if self.wave_number > self.MAX_WAVES:
                    self.game_over = True # VICTORY
                else:
                    self.game_phase = "PLACEMENT"
                    self._update_available_spores()
                    self.selected_portal_type_index = 0
                    # sfx: wave_complete_fanfare
    
        terminated = self.fortress_health <= 0 or self.game_over or self.steps >= self.MAX_STEPS
        
        if terminated:
            if self.fortress_health <= 0:
                reward = -100 # Loss penalty
            elif self.wave_number > self.MAX_WAVES:
                reward = 100 # Win bonus
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_path()
        self._render_portal_spots()
        self._render_fortress()
        self._render_particles()
        self._render_slugs()
        self._render_portals()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "fortress_health": self.fortress_health,
            "slugs_remaining": len(self.slugs),
            "game_phase": self.game_phase,
        }

    # --- Update Logic ---
    
    def _update_portals(self):
        reward = 0
        for portal in self.portals:
            portal['cooldown'] = max(0, portal['cooldown'] - 1)
            portal['animation_angle'] += 0.05
            if portal['cooldown'] == 0:
                if portal['type'] == 'explosive' and self.slugs:
                    portal['cooldown'] = 120 # Fire rate
                    target_slug = min(self.slugs, key=lambda s: np.linalg.norm(s['pos'] - portal['pos']))
                    direction = target_slug['pos'] - portal['pos']
                    vel = direction / np.linalg.norm(direction) * 3
                    self.particles.append(self._create_spore(portal['pos'], vel, 'explosive'))
                    # sfx: explosive_spore_fire
                elif portal['type'] == 'healing':
                    portal['cooldown'] = 180
                    direction = self.fortress_pos - portal['pos']
                    vel = direction / np.linalg.norm(direction) * 2
                    self.particles.append(self._create_spore(portal['pos'], vel, 'healing'))
                    # sfx: healing_spore_fire
                elif portal['type'] == 'slowing' and self.slugs:
                    portal['cooldown'] = 90
                    target_slug = min(self.slugs, key=lambda s: np.linalg.norm(s['pos'] - portal['pos']))
                    direction = target_slug['pos'] - portal['pos']
                    vel = direction / np.linalg.norm(direction) * 4
                    self.particles.append(self._create_spore(portal['pos'], vel, 'slowing'))
                    # sfx: slowing_spore_fire
        return reward

    def _update_slugs(self):
        reward = 0
        for slug in reversed(self.slugs):
            # Movement
            path_idx = slug['path_index']
            if path_idx >= len(self.path) - 1:
                # Reached fortress
                self.fortress_health = max(0, self.fortress_health - slug['health'])
                reward -= 0.5 * slug['health'] # Fortress damage penalty
                self.slugs.remove(slug)
                self.fortress_hit_flash = 10
                # sfx: fortress_hit_damage
                continue

            start_node = self.path[path_idx]
            end_node = self.path[path_idx + 1]
            segment_vec = end_node - start_node
            segment_len = np.linalg.norm(segment_vec)
            
            if segment_len > 0:
                move_dist = slug['speed'] * slug['speed_modifier']
                slug['t'] += move_dist / segment_len
                if slug['t'] >= 1.0:
                    slug['path_index'] += 1
                    slug['t'] = 0
                slug['pos'] = start_node + (end_node - start_node) * slug['t']
            
            # Slow effect decay
            slug['slow_timer'] = max(0, slug['slow_timer'] - 1)
            if slug['slow_timer'] == 0:
                slug['speed_modifier'] = 1.0
        
        self.wave_timer = max(0, self.wave_timer - 1)
        return reward

    def _update_particles(self):
        reward = 0
        for p in reversed(self.particles):
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                if p['type'] == 'spore_explosive': # Detonate
                    self._create_explosion(p['pos'], 30, 5)
                    # sfx: explosion
                self.particles.remove(p)
                continue

            # Spore collision checks
            if p['type'].startswith('spore_'):
                if p['type'] == 'spore_healing':
                    if np.linalg.norm(p['pos'] - self.fortress_pos) < 20:
                        self.fortress_health = min(self.FORTRESS_MAX_HEALTH, self.fortress_health + 2)
                        # sfx: fortress_heal
                        self.particles.remove(p)
                else: # Damaging/slowing spores
                    for slug in self.slugs:
                        if np.linalg.norm(p['pos'] - slug['pos']) < 10:
                            if p['type'] == 'spore_explosive':
                                self._create_explosion(p['pos'], 40, 5)
                                # sfx: explosion
                                reward += self._damage_slugs_in_radius(p['pos'], 40, 5)
                            elif p['type'] == 'spore_slowing':
                                slug['slow_timer'] = 150 # 5 seconds at 30fps
                                slug['speed_modifier'] = 0.5
                                reward += 0.1 # Small reward for slowing
                                self._create_effect_burst(p['pos'], self.SPORE_TYPES['slowing']['color'], 10)
                                # sfx: slow_hit
                            self.particles.remove(p)
                            break
        return reward

    # --- Spawning and Generation ---

    def _spawn_wave(self):
        num_slugs = 5 + self.wave_number
        base_health = 5 + (self.wave_number // 2)
        base_speed = 0.5 + (self.wave_number * 0.05)
        
        self.wave_timer = (num_slugs * 30) + 120 # Time before slugs start spawning
        
        for i in range(num_slugs):
            self.slugs.append({
                'pos': np.copy(self.path[0]),
                'health': base_health,
                'speed': base_speed * self.np_random.uniform(0.9, 1.1),
                'path_index': 0,
                't': 0.0,
                'spawn_delay': i * 30, # Staggered spawn
                'hit_flash': 0,
                'slow_timer': 0,
                'speed_modifier': 1.0,
            })
    
    def _generate_path(self):
        self.path = []
        start_y = self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
        self.path.append(np.array([self.SCREEN_WIDTH - 20, start_y]))
        
        num_waypoints = 4
        current_x = self.SCREEN_WIDTH - 120
        for i in range(num_waypoints):
            y = self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
            self.path.append(np.array([current_x, y]))
            current_x -= self.np_random.uniform(80, 120)
        
        self.path.append(self.fortress_pos)

    def _generate_portal_spots(self):
        self.portal_spots = []
        # Create a grid of potential spots
        for i in range(4):
            for j in range(3):
                x = 200 + i * 100
                y = 80 + j * 120
                # Ensure spots are not too close to the path
                min_dist_to_path = min(np.linalg.norm(np.array([x,y]) - p) for p in self.path)
                if min_dist_to_path > 40:
                    self.portal_spots.append(np.array([x, y]))
        # Sort them for consistent cursor movement
        self.portal_spots.sort(key=lambda p: (p[1], p[0]))

    def _update_available_spores(self):
        self.available_spore_types = []
        for i, (key, props) in enumerate(self.SPORE_TYPES.items()):
            if self.wave_number >= props['unlock_wave']:
                self.available_spore_types.append(i)

    # --- Particle and Effect Creation ---
    
    def _create_spore(self, pos, vel, spore_type_key):
        return {
            'pos': np.copy(pos), 'vel': vel, 'lifespan': 150,
            'color': self.SPORE_TYPES[spore_type_key]['color'],
            'type': f'spore_{spore_type_key}'
        }

    def _create_explosion(self, pos, radius, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': np.copy(pos), 'vel': vel, 'lifespan': 20,
                'color': (255, self.np_random.integers(100, 200), 0),
                'type': 'explosion_spark'
            })

    def _create_effect_burst(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': np.copy(pos), 'vel': vel, 'lifespan': 15,
                'color': color, 'type': 'effect_spark'
            })

    def _damage_slugs_in_radius(self, center, radius, damage):
        reward = 0
        slugs_to_remove = []
        for slug in self.slugs:
            if np.linalg.norm(slug['pos'] - center) < radius:
                slug['health'] -= damage
                slug['hit_flash'] = 5
                reward += 0.1 # Reward for hitting a slug
                if slug['health'] <= 0:
                    # sfx: slug_die
                    slugs_to_remove.append(slug)
        
        if slugs_to_remove:
            self.slugs = [s for s in self.slugs if s not in slugs_to_remove]
        return reward

    # --- Rendering ---

    def _render_fortress(self):
        pos = self.fortress_pos.astype(int)
        # Base
        pygame.draw.rect(self.screen, self.COLOR_FORTRESS, (pos[0] - 15, pos[1], 30, 20))
        # Cap with glow
        for i in range(10, 0, -1):
            alpha = int(150 * (1 - i / 10))
            color = (*self.COLOR_FORTRESS_CAP, alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 20 + i, color)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 20, self.COLOR_FORTRESS_CAP)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 20, self.COLOR_FORTRESS_CAP)
        
        # Red flash on hit
        if self.fortress_hit_flash > 0:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.fortress_hit_flash / 10.0))
            s.fill((*self.COLOR_FORTRESS_DAMAGE, alpha))
            self.screen.blit(s, (0,0))

    def _render_path(self):
        if len(self.path) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_PATH, False, self.path, 3)

    def _render_portal_spots(self):
        for spot in self.portal_spots:
            is_occupied = any(np.array_equal(p['pos'], spot) for p in self.portals)
            if not is_occupied:
                pygame.gfxdraw.aacircle(self.screen, int(spot[0]), int(spot[1]), 15, self.COLOR_PORTAL_SPOT)

    def _render_portals(self):
        for portal in self.portals:
            pos = portal['pos'].astype(int)
            color = self.SPORE_TYPES[portal['type']]['color']
            # Glowing effect
            for i in range(10, 0, -1):
                alpha = int(100 * (1 - i / 10))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15 + i, (*color, alpha))
            
            # Inner rotating dots
            for i in range(3):
                angle = portal['animation_angle'] + i * 2 * math.pi / 3
                dot_pos = pos + np.array([math.cos(angle), math.sin(angle)]) * 10
                pygame.gfxdraw.filled_circle(self.screen, int(dot_pos[0]), int(dot_pos[1]), 2, color)

    def _render_slugs(self):
        for slug in self.slugs:
            if slug.get('spawn_delay', 0) > 0:
                slug['spawn_delay'] -= 1
                continue
            
            pos = slug['pos'].astype(int)
            color = self.COLOR_SLUG_HIT if slug['hit_flash'] > 0 else self.COLOR_SLUG
            if slug['hit_flash'] > 0: slug['hit_flash'] -= 1
            
            # Simple body animation
            body_len = 12
            target_node_idx = min(slug['path_index'] + 1, len(self.path) - 1)
            angle = math.atan2(self.path[target_node_idx][1] - slug['pos'][1], self.path[target_node_idx][0] - slug['pos'][0])

            for i in range(body_len):
                anim_offset = math.sin(self.steps * 0.2 + i * 0.5) * 2
                p_pos = pos - np.array([math.cos(angle), math.sin(angle)]) * i
                p_pos += np.array([math.sin(angle), -math.cos(angle)]) * anim_offset
                size = max(1, 6 - i // 2)
                
                # Slow effect visual
                slug_color = color
                if slug['slow_timer'] > 0:
                    c1 = np.array(color)
                    c2 = np.array(self.SPORE_TYPES['slowing']['color'])
                    mix = (math.sin(self.steps * 0.3) + 1) / 2
                    slug_color = tuple((c1 * (1-mix) + c2 * mix).astype(int))

                pygame.gfxdraw.filled_circle(self.screen, int(p_pos[0]), int(p_pos[1]), size, slug_color)

    def _render_particles(self):
        for p in self.particles:
            pos = p['pos'].astype(int)
            if p['type'].startswith('spore_'):
                # Spore with tail
                tail_pos = (p['pos'] - p['vel'] * 2).astype(int)
                pygame.draw.aaline(self.screen, p['color'], pos, tail_pos, 2)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, p['color'])
            else: # Sparks
                size = int(p['lifespan'] / 5)
                if size > 0:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, p['color'])

    def _render_ui(self):
        # --- Fortress Health Bar ---
        health_percent = self.fortress_health / self.FORTRESS_MAX_HEALTH
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_FORTRESS_DAMAGE, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_FORTRESS_HEALTH, (10, 10, int(bar_width * health_percent), 20))
        health_text = self.font_small.render(f"FORTRESS: {int(self.fortress_health)}/{self.FORTRESS_MAX_HEALTH}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # --- Wave Info ---
        wave_text = self.font_large.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # --- Phase-specific UI ---
        if self.game_phase == "PLACEMENT":
            # Cursor
            cursor_pos = self.portal_spots[self.cursor_spot_index]
            pygame.gfxdraw.aacircle(self.screen, int(cursor_pos[0]), int(cursor_pos[1]), 20, self.COLOR_CURSOR)
            
            # Spore selection UI
            if self.available_spore_types:
                spore_idx = self.available_spore_types[self.selected_portal_type_index]
                spore_key = list(self.SPORE_TYPES.keys())[spore_idx]
                spore_props = self.SPORE_TYPES[spore_key]
                
                spore_text = self.font_small.render(f"Next Portal: {spore_props['name']}", True, self.COLOR_UI_TEXT)
                self.screen.blit(spore_text, (10, self.SCREEN_HEIGHT - 30))
                pygame.gfxdraw.filled_circle(self.screen, 160, self.SCREEN_HEIGHT - 22, 8, spore_props['color'])
            
            # Prompt
            prompt_text = self.font_large.render("Place Portals (Arrows, Space). Wait to start wave.", True, self.COLOR_UI_TEXT)
            self.screen.blit(prompt_text, (self.SCREEN_WIDTH/2 - prompt_text.get_width()/2, self.SCREEN_HEIGHT - 40))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run when the environment is used by the training pipeline
    # but is useful for testing and debugging.
    
    # Un-comment the line below to run with a graphical display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fungal Fortress Defense")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0

    # Game loop
    while not terminated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")

    env.close()