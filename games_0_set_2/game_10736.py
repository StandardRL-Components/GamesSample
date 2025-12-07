import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:51:55.850773
# Source Brief: brief_00736.md
# Brief Index: 736
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Race down a DNA helix, collecting matching base pairs and splicing genes for power-ups "
        "while avoiding obstacles and sequence degradation."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to move between lanes. Press space to splice genes and "
        "gain mutations. Use shift to activate special abilities."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.LANE_COUNT = 4
        self.LANE_WIDTH = 60
        self.TOTAL_LANE_WIDTH = self.LANE_COUNT * self.LANE_WIDTH
        self.LANE_START_X = (self.WIDTH - self.TOTAL_LANE_WIDTH) / 2

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_HELIX_BG = (20, 35, 65)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_OBSTACLE = (60, 60, 80)
        self.COLOR_OBSTACLE_GLOW = (180, 50, 50)
        self.COLOR_SPLICER = (255, 180, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.BASE_PAIR_COLORS = {
            'A': (100, 255, 100), 'T': (255, 100, 100),
            'C': (100, 100, 255), 'G': (255, 255, 100)
        }
        self.BASE_PAIR_TYPES = list(self.BASE_PAIR_COLORS.keys())

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_mutation = pygame.font.SysFont("Consolas", 16)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # Game state variables are initialized in reset()
        self.player_lane = 0
        self.player_visual_x = 0
        self.player_radius = 0
        self.player_squash = 0
        self.scroll_speed = 0
        self.base_pair_spawn_prob = 0
        self.obstacle_spawn_prob = 0
        self.splicer_spawn_prob = 0
        self.timer = 0
        self.progress = 0
        self.total_length = 0
        self.entities = []
        self.particles = []
        self.active_mutations = {}
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.terminated = False

        # Player state
        self.player_lane = 1
        self.player_visual_x = self.LANE_START_X + self.LANE_WIDTH * (self.player_lane + 0.5)
        self.player_radius = 12
        self.player_squash = 1.0 # For visual effect

        # World state
        self.scroll_speed = 2.0
        self.base_pair_spawn_prob = 0.05
        self.obstacle_spawn_prob = 0.01
        self.splicer_spawn_prob = 0.005
        self.timer = 1000
        self.progress = 0
        self.total_length = 5000

        # Entity lists
        self.entities = [] # {'type': 'base'|'obstacle'|'splicer', 'pos': [x,y], 'lane': i, 'data':{}}
        self.particles = [] # {'pos': [x,y], 'vel': [vx,vy], 'radius': r, 'color': c, 'life': l}

        # Mutations
        self.active_mutations = {} # {'name': duration}
        self.mutation_definitions = {
            'hyper_speed': {'duration': 200, 'type': 'good'},
            'obstacle_immunity': {'duration': 150, 'type': 'good'},
            'phase_shift_ability': {'duration': float('inf'), 'type': 'good'},
            'inverted_controls': {'duration': 300, 'type': 'bad'},
            'bloat': {'duration': 250, 'type': 'bad'},
        }

        # Input state
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.timer -= 1
        self.progress += self.scroll_speed

        # --- 1. Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # Apply inverted controls mutation
        if 'inverted_controls' in self.active_mutations:
            if movement == 3: movement = 4
            elif movement == 4: movement = 3

        if movement == 3 and self.player_lane > 0: # Left
            self.player_lane -= 1
            self.player_squash = 1.4 # Squash and stretch effect
        if movement == 4 and self.player_lane < self.LANE_COUNT - 1: # Right
            self.player_lane += 1
            self.player_squash = 1.4

        # --- 2. Update Game Logic ---
        self._update_difficulty()
        self._update_mutations()
        reward += self._handle_actions(space_pressed, shift_pressed)
        self._spawn_entities()
        self._update_entities()
        reward += self._check_interactions()
        self._update_particles()
        self._update_player_visuals()

        # --- 3. Check Termination & Final Reward ---
        self.terminated = self._is_terminated()
        if self.terminated:
            if self.progress >= self.total_length:
                reward += 100 # Win
            elif self.timer <= 0:
                reward -= 50 # Timeout
            else: # Must be a collision
                reward -= 10

        self.score += reward
        return self._get_observation(), reward, self.terminated, False, self._get_info()

    def _update_difficulty(self):
        self.scroll_speed = 2.0 + (self.steps / 250) * 0.5
        if 'hyper_speed' in self.active_mutations:
            self.scroll_speed *= 1.5
        self.obstacle_spawn_prob = min(0.1, 0.01 + (self.steps / 100) * 0.005)

    def _update_mutations(self):
        expired = [name for name, duration in self.active_mutations.items() if (duration - 1) <= 0]
        for name in expired:
            if name != 'phase_shift_ability': # This ability is permanent until used
                del self.active_mutations[name]
        for name in self.active_mutations:
            if name != 'phase_shift_ability':
                self.active_mutations[name] -= 1

    def _handle_actions(self, space_pressed, shift_pressed):
        reward = 0
        player_y = self.HEIGHT * 0.8

        if space_pressed:
            closest_splicer = None
            min_dist = float('inf')
            for e in self.entities:
                if e['type'] == 'splicer':
                    dist = abs(e['pos'][1] - player_y) + abs(e['lane'] - self.player_lane) * self.LANE_WIDTH
                    if dist < 60 and dist < min_dist:
                        min_dist = dist
                        closest_splicer = e
            if closest_splicer:
                self.entities.remove(closest_splicer)
                reward += 1
                self._apply_random_mutation()
                self._create_particle_burst(
                    [self.player_visual_x, player_y], 30, self.COLOR_SPLICER, 5, 40
                )
        
        if shift_pressed and 'phase_shift_ability' in self.active_mutations:
            del self.active_mutations['phase_shift_ability']
            self.active_mutations['obstacle_immunity'] = 90 # 3 seconds of immunity
            self._create_particle_burst(
                [self.player_visual_x, player_y], 20, self.COLOR_PLAYER_GLOW, 3, 30, vel_mult=2
            )
        return reward

    def _apply_random_mutation(self):
        mutation_name = self.np_random.choice(list(self.mutation_definitions.keys()))
        mutation = self.mutation_definitions[mutation_name]
        self.active_mutations[mutation_name] = mutation['duration']
        
        if mutation['type'] == 'good':
            return 5
        else:
            return -2

    def _spawn_entities(self):
        occupied_lanes = {e['lane'] for e in self.entities if e['pos'][1] < 50}
        available_lanes = [i for i in range(self.LANE_COUNT) if i not in occupied_lanes]
        if not available_lanes: return

        lane = self.np_random.choice(available_lanes)
        spawn_roll = self.np_random.random()

        if spawn_roll < self.obstacle_spawn_prob:
            self.entities.append({'type': 'obstacle', 'pos': [0, -20], 'lane': lane, 'data': {}})
        elif spawn_roll < self.obstacle_spawn_prob + self.splicer_spawn_prob:
            self.entities.append({'type': 'splicer', 'pos': [0, -20], 'lane': lane, 'data': {}})
        elif spawn_roll < self.obstacle_spawn_prob + self.splicer_spawn_prob + self.base_pair_spawn_prob:
            base_type = self.np_random.choice(self.BASE_PAIR_TYPES)
            self.entities.append({'type': 'base', 'pos': [0, -20], 'lane': lane, 'data': {'base': base_type}})

    def _update_entities(self):
        for e in self.entities:
            e['pos'][1] += self.scroll_speed
            e['pos'][0] = self.LANE_START_X + self.LANE_WIDTH * (e['lane'] + 0.5)
        self.entities = [e for e in self.entities if e['pos'][1] < self.HEIGHT + 50]

    def _check_interactions(self):
        reward = 0
        player_y = self.HEIGHT * 0.8
        player_radius = self.player_radius * (1.5 if 'bloat' in self.active_mutations else 1.0)
        
        entities_to_remove = []
        for e in self.entities:
            entity_pos = e['pos']
            dist_y = abs(entity_pos[1] - player_y)

            if e['lane'] == self.player_lane:
                if dist_y < player_radius:
                    if e['type'] == 'base':
                        reward += 0.1
                        self._create_particle_burst(entity_pos, 10, self.BASE_PAIR_COLORS[e['data']['base']], 2, 20)
                        entities_to_remove.append(e)
                    elif e['type'] == 'obstacle' and 'obstacle_immunity' not in self.active_mutations:
                        self.terminated = True
                        self._create_particle_burst(entity_pos, 50, self.COLOR_OBSTACLE_GLOW, 8, 60, vel_mult=5)
                elif e['type'] == 'base' and entity_pos[1] > player_y + player_radius:
                    reward -= 0.1
                    entities_to_remove.append(e)
        
        self.entities = [e for e in self.entities if e not in entities_to_remove]
        return reward

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] *= 0.95
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0.5]
    
    def _update_player_visuals(self):
        target_x = self.LANE_START_X + self.LANE_WIDTH * (self.player_lane + 0.5)
        self.player_visual_x += (target_x - self.player_visual_x) * 0.4
        self.player_squash = max(1.0, self.player_squash * 0.85)

    def _is_terminated(self):
        return self.terminated or self.timer <= 0 or self.progress >= self.total_length

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_helix_background()
        self._render_entities()
        self._render_particles()
        if not self.terminated:
            self._render_player()

    def _render_helix_background(self):
        num_rungs = 25
        scroll_offset = self.progress % (self.HEIGHT / num_rungs * 2)

        for i in range(num_rungs + 2):
            y = i * (self.HEIGHT / num_rungs) - scroll_offset
            amplitude = self.TOTAL_LANE_WIDTH / 2 - 10
            center = self.WIDTH / 2
            
            x1 = center + amplitude * math.sin(y * 0.05 + self.progress * 0.01)
            x2 = center - amplitude * math.sin(y * 0.05 + self.progress * 0.01)
            
            pygame.draw.line(self.screen, self.COLOR_HELIX_BG, (int(x1), int(y)), (int(x2), int(y)), 2)
            pygame.gfxdraw.filled_circle(self.screen, int(x1), int(y), 4, self.COLOR_HELIX_BG)
            pygame.gfxdraw.filled_circle(self.screen, int(x2), int(y), 4, self.COLOR_HELIX_BG)

    def _render_entities(self):
        for e in self.entities:
            pos = (int(e['pos'][0]), int(e['pos'][1]))
            if e['type'] == 'base':
                color = self.BASE_PAIR_COLORS[e['data']['base']]
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, color)
            elif e['type'] == 'obstacle':
                p = [
                    (pos[0], pos[1] - 12), (pos[0] + 12, pos[1]),
                    (pos[0], pos[1] + 12), (pos[0] - 12, pos[1])
                ]
                pygame.gfxdraw.aapolygon(self.screen, p, self.COLOR_OBSTACLE_GLOW)
                pygame.gfxdraw.filled_polygon(self.screen, p, self.COLOR_OBSTACLE)
            elif e['type'] == 'splicer':
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_SPLICER)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, (255,255,255))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(8 + 3 * math.sin(self.steps * 0.2)), self.COLOR_SPLICER)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                alpha = int(255 * (p['life'] / p['max_life']))
                color_tuple = p['color']
                if len(color_tuple) == 3:
                    color = (*color_tuple, alpha)
                else: # Already has alpha
                    color = (color_tuple[0], color_tuple[1], color_tuple[2], alpha)
                
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))


    def _render_player(self):
        y = self.HEIGHT * 0.8
        radius_x = self.player_radius * self.player_squash
        radius_y = self.player_radius / self.player_squash
        if 'bloat' in self.active_mutations:
            radius_x *= 1.5
            radius_y *= 1.5

        rect = pygame.Rect(0, 0, int(radius_x * 2.5), int(radius_y * 2.5))
        rect.center = (int(self.player_visual_x), int(y))
        
        # Glow effect
        glow_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        glow_color = (100, 255, 100, 128) if 'obstacle_immunity' in self.active_mutations else (*self.COLOR_PLAYER_GLOW, 100)
        pygame.draw.ellipse(glow_surf, glow_color, glow_surf.get_rect())
        self.screen.blit(glow_surf, rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)

        # Player body
        player_rect = pygame.Rect(0, 0, int(radius_x * 2), int(radius_y * 2))
        player_rect.center = (int(self.player_visual_x), int(y))
        pygame.draw.ellipse(self.screen, self.COLOR_PLAYER, player_rect)

    def _render_ui(self):
        # Top Left: Speed
        speed_text = self.font_ui.render(f"SPEED: {self.scroll_speed:.1f}x", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (10, 10))

        # Top Right: Timer
        timer_text = self.font_ui.render(f"TIMER: {self.timer}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Bottom Right: Progress
        progress_perc = min(1.0, self.progress / self.total_length) * 100
        progress_text = self.font_ui.render(f"TELOMERE: {progress_perc:.1f}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(progress_text, (self.WIDTH - progress_text.get_width() - 10, self.HEIGHT - progress_text.get_height() - 10))

        # Bottom Left: Mutations
        mut_y = self.HEIGHT - 20
        for name, duration in self.active_mutations.items():
            is_bad = self.mutation_definitions[name]['type'] == 'bad'
            color = (255, 100, 100) if is_bad else (100, 255, 100)
            duration_str = f"{duration/self.FPS:.1f}s" if duration != float('inf') else "READY"
            mut_text = self.font_mutation.render(f"{name.upper()}: {duration_str}", True, color)
            self.screen.blit(mut_text, (10, mut_y - mut_text.get_height()))
            mut_y -= mut_text.get_height() + 2

        # Game Over Text
        if self.terminated:
            msg = "SEQUENCE COMPLETE" if self.progress >= self.total_length else "SEQUENCE FAILED"
            color = (100, 255, 100) if self.progress >= self.total_length else (255, 100, 100)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "progress": self.progress, "timer": self.timer}

    def _create_particle_burst(self, pos, count, color, radius, life, vel_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * vel_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'radius': radius * self.np_random.uniform(0.5, 1.2),
                'color': color, 'life': life * self.np_random.uniform(0.7, 1.3), 'max_life': life
            })

# Example usage to run and visualize the game
if __name__ == '__main__':
    # Un-comment the line below to run with a display
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # If not running headless, create a display
    if "SDL_VIDEODRIVER" not in os.environ:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("DNA Helix Racer")
    else:
        screen = None

    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    
    while running:
        movement, space, shift = 0, 0, 0
        
        # Only process events if a screen exists
        if screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            if terminated and keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
        else: # In headless mode, just run for a fixed number of steps
            if env.steps > 2000:
                running = False

        action = [movement, space, shift]
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)

        # Draw the observation from the environment to the screen if it exists
        if screen:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(env.FPS)
        
    pygame.quit()