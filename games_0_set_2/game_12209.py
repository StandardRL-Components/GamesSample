import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:16:00.233643
# Source Brief: brief_02209.md
# Brief Index: 2209
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Navigate a shifting fractal landscape using time manipulation and portal travel
    to reach the fractal's core.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a shifting fractal landscape by jumping through portals. Use time manipulation to reveal hidden paths and reach the core before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select a portal. Press space to travel through the selected portal. Press shift to use a time charge, revealing hidden portals."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    FPS = 30
    
    # --- Colors ---
    COLOR_BG_START = (5, 10, 20)
    COLOR_BG_END = (20, 5, 30)
    COLOR_FRACTAL_LINE = (100, 120, 150)
    COLOR_PORTAL = (0, 150, 255)
    COLOR_PORTAL_HIDDEN = (70, 80, 140)
    COLOR_PORTAL_SELECTED = (50, 255, 150)
    COLOR_CORE = (255, 200, 0)
    COLOR_TIME_DISTORT = (120, 50, 200, 100) # RGBA
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BAR_BG = (50, 50, 80)
    COLOR_UI_BAR_FG = (100, 180, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 36, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.fractal_level = 1
        self.time_charges = 0
        self.fractal_graph = {}
        self.core_id = -1
        self.current_section_id = -1
        self.cursor_index = 0
        self.time_manipulation_active = False
        self.time_manipulation_timer = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.max_dist_to_core = 1.0
        self.previous_dist_to_core = 1.0

        # --- Initial Reset ---
        # Note: We don't call reset() here because the user of the env does that.
        # We do, however, need to initialize the state variables.
        self._initialize_state_vars()

    def _initialize_state_vars(self):
        """Initializes or re-initializes all state variables to default values."""
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_charges = 5
        self.fractal_graph = {}
        self.core_id = -1
        self.current_section_id = -1
        self.cursor_index = 0
        self.time_manipulation_active = False
        self.time_manipulation_timer = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.max_dist_to_core = 1.0
        self.previous_dist_to_core = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state_vars()

        # --- Level setup based on progression ---
        if options and "level" in options:
            self.fractal_level = options.get("level", 1)
        
        self.time_charges = max(1, 5 - (self.fractal_level - 1) // 3)
        fractal_depth = 3 + (self.fractal_level - 1)
        
        # --- Generate Fractal ---
        self._generate_fractal(depth=fractal_depth)
        
        if self.current_section_id != -1: # Check if fractal generation was successful
            self.max_dist_to_core = self._get_distance_to_core(self.current_section_id)
            self.previous_dist_to_core = self.max_dist_to_core
        else: # Handle edge case of empty fractal
            self.max_dist_to_core = 1.0
            self.previous_dist_to_core = 1.0


        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Unpack Actions and Handle Input ---
        movement, space_held, shift_held = action
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # --- Update Game Logic ---
        self._update_time_manipulation()
        
        # 1. Handle Time Manipulation (Shift)
        if shift_pressed and self.time_charges > 0 and not self.time_manipulation_active:
            # SFX: Time-warp activate
            self.time_charges -= 1
            self.time_manipulation_active = True
            self.time_manipulation_timer = self.FPS * 4  # 4 seconds
            self._create_particles(self.screen.get_rect().center, 50, (120, 50, 200), 2, 4)
            
            # Reward for revealing new portals
            current_portals = self.fractal_graph[self.current_section_id]['portals']
            if any(p['hidden'] for p in current_portals):
                reward += 1.0

        # 2. Handle Cursor Movement
        if movement != 0:
            self._move_cursor(movement)

        # 3. Handle Portal Activation (Space)
        if space_pressed:
            reward += self._activate_portal()
        
        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.current_section_id == self.core_id:
            reward += 100.0
            self.score += reward
            self.game_over = True
            terminated = True
            # Advance level for next reset
            self.fractal_level += 1
        elif self.steps >= self.MAX_STEPS:
            reward -= 100.0 # Penalty for timeout
            self.score += reward
            self.game_over = True
            truncated = True # Use truncated for time limit
            terminated = True # Gymnasium standard is to set both
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _get_observation(self):
        self._draw_gradient_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.fractal_level,
            "time_charges": self.time_charges,
            "distance_to_core": self._get_distance_to_core(self.current_section_id)
        }
        
    # --- Game Logic Helpers ---
    def _generate_fractal(self, depth):
        self.fractal_graph = {}
        node_id_counter = 0
        
        # Create core node
        self.core_id = node_id_counter
        self.fractal_graph[self.core_id] = {
            'id': self.core_id, 'depth': 0, 'pos': (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2),
            'portals': [], 'parent': None
        }
        node_id_counter += 1
        
        queue = deque([self.core_id])
        leaf_nodes = []

        while queue:
            parent_id = queue.popleft()
            parent_node = self.fractal_graph[parent_id]
            
            if parent_node['depth'] >= depth:
                if parent_id != self.core_id:
                    leaf_nodes.append(parent_id)
                continue

            num_children = self.np_random.integers(2, 4)
            for i in range(num_children):
                child_id = node_id_counter
                angle = self.np_random.uniform(0, 2 * math.pi)
                radius = 100 + self.np_random.uniform(-20, 20)
                
                # Position relative to parent, but constrained to screen
                pos_x = parent_node['pos'][0] + math.cos(angle) * radius
                pos_y = parent_node['pos'][1] + math.sin(angle) * radius
                pos_x = np.clip(pos_x, 50, self.SCREEN_WIDTH - 50)
                pos_y = np.clip(pos_y, 50, self.SCREEN_HEIGHT - 50)

                self.fractal_graph[child_id] = {
                    'id': child_id, 'depth': parent_node['depth'] + 1, 'pos': (pos_x, pos_y),
                    'portals': [], 'parent': parent_id
                }
                node_id_counter += 1
                queue.append(child_id)
        
        # Create portals (connections)
        for node_id, node in self.fractal_graph.items():
            # Connect to parent
            if node['parent'] is not None:
                node['portals'].append({'target_id': node['parent'], 'hidden': False})
            # Connect to children
            children = [cid for cid, cnode in self.fractal_graph.items() if cnode.get('parent') == node_id]
            for child_id in children:
                # 20% chance of being a hidden portal
                is_hidden = self.np_random.random() < 0.20
                node['portals'].append({'target_id': child_id, 'hidden': is_hidden})
        
        # Set start position
        if not leaf_nodes: # Handle shallow fractals
            potentials = [nid for nid, n in self.fractal_graph.items() if nid != self.core_id]
            if potentials:
                self.current_section_id = self.np_random.choice(potentials)
            else: # Only core exists
                self.current_section_id = self.core_id
        else:
            self.current_section_id = self.np_random.choice(leaf_nodes)

        # Ensure start node has at least one non-hidden path
        if self.current_section_id != -1 and self.current_section_id in self.fractal_graph:
            start_node = self.fractal_graph[self.current_section_id]
            if all(p['hidden'] for p in start_node['portals']) and start_node['portals']:
                start_node['portals'][0]['hidden'] = False
        
        self.cursor_index = 0

    def _update_time_manipulation(self):
        if self.time_manipulation_active:
            self.time_manipulation_timer -= 1
            if self.time_manipulation_timer <= 0:
                self.time_manipulation_active = False
                self.time_manipulation_timer = 0
                # SFX: Time-warp fade
    
    def _move_cursor(self, direction):
        current_section = self.fractal_graph.get(self.current_section_id)
        if not current_section or not current_section['portals']:
            return

        portals = current_section['portals']
        num_portals = len(portals)
        if num_portals <= 1:
            return

        current_portal_pos = self.fractal_graph[portals[self.cursor_index]['target_id']]['pos']
        
        best_candidate = -1
        min_cost = float('inf')

        for i in range(num_portals):
            if i == self.cursor_index:
                continue
            
            candidate_pos = self.fractal_graph[portals[i]['target_id']]['pos']
            dx = candidate_pos[0] - current_portal_pos[0]
            dy = candidate_pos[1] - current_portal_pos[1]
            
            if dx == 0 and dy == 0: continue

            angle = math.atan2(-dy, dx) # -dy because pygame y is inverted
            dist = math.sqrt(dx**2 + dy**2)

            is_match = False
            if direction == 1: # Up
                is_match = math.pi / 4 < angle < 3 * math.pi / 4
            elif direction == 2: # Down
                is_match = -3 * math.pi / 4 < angle < -math.pi / 4
            elif direction == 3: # Left
                is_match = (3 * math.pi / 4 < angle <= math.pi) or (-math.pi <= angle < -3 * math.pi / 4)
            elif direction == 4: # Right
                is_match = -math.pi / 4 < angle < math.pi / 4
            
            if is_match and dist < min_cost:
                min_cost = dist
                best_candidate = i
        
        if best_candidate != -1:
            self.cursor_index = best_candidate
            # SFX: Cursor move
    
    def _activate_portal(self):
        current_section = self.fractal_graph.get(self.current_section_id)
        if not current_section or not current_section['portals']:
            return 0.0

        portals = current_section['portals']
        if self.cursor_index >= len(portals):
            return 0.0

        selected_portal = portals[self.cursor_index]
        
        # Cannot use hidden portals unless time manipulation is active
        if selected_portal['hidden'] and not self.time_manipulation_active:
            # SFX: Action failed
            return 0.0

        # SFX: Portal whoosh
        self.previous_dist_to_core = self._get_distance_to_core(self.current_section_id)
        
        # Teleport
        start_pos = current_section['pos']
        self.current_section_id = selected_portal['target_id']
        end_pos = self.fractal_graph[self.current_section_id]['pos']
        self._create_particles(start_pos, 20, self.COLOR_PORTAL_SELECTED, 1, 3, end_pos)

        # Reset cursor and time effect
        self.cursor_index = 0
        self.time_manipulation_active = False
        self.time_manipulation_timer = 0
        
        # Calculate reward
        new_dist_to_core = self._get_distance_to_core(self.current_section_id)
        if new_dist_to_core < self.previous_dist_to_core:
            return 0.1
        elif new_dist_to_core > self.previous_dist_to_core:
            return -0.1
        return 0.0
        
    def _get_distance_to_core(self, section_id):
        if section_id == -1 or self.core_id == -1: return 0.0
        pos1 = self.fractal_graph[section_id]['pos']
        pos2 = self.fractal_graph[self.core_id]['pos']
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    # --- Rendering Helpers ---
    def _render_game(self):
        # Update and draw particles
        self._update_and_draw_particles()

        current_node = self.fractal_graph.get(self.current_section_id)
        if not current_node: return

        # Draw core marker
        core_pos = self.fractal_graph[self.core_id]['pos']
        self._draw_glowing_circle(core_pos, 20, self.COLOR_CORE, 0.5 + 0.5 * math.sin(self.steps * 0.1))

        # Draw connections and portals
        for i, portal in enumerate(current_node['portals']):
            target_node = self.fractal_graph[portal['target_id']]
            is_visible = not portal['hidden'] or self.time_manipulation_active
            is_selected = (i == self.cursor_index)
            
            if not is_visible: continue

            # Draw line from center to portal location
            pygame.draw.aaline(self.screen, self.COLOR_FRACTAL_LINE, current_node['pos'], target_node['pos'], 1)

            # Draw portal
            color = self.COLOR_PORTAL
            if portal['hidden']:
                color = self.COLOR_PORTAL_HIDDEN
            if is_selected:
                color = self.COLOR_PORTAL_SELECTED
            
            glow = 0.0
            if is_selected:
                glow = 0.6 + 0.4 * math.sin(self.steps * 0.2)
            elif portal['hidden']:
                glow = 0.4 + 0.3 * math.sin(self.steps * 0.15 + i)

            self._draw_glowing_circle(target_node['pos'], 10, color, glow)
        
        # Draw time distortion effect
        if self.time_manipulation_active:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = self.COLOR_TIME_DISTORT[3] * min(1.0, self.time_manipulation_timer / (self.FPS * 0.5))
            overlay.fill((*self.COLOR_TIME_DISTORT[:3], alpha))
            self.screen.blit(overlay, (0, 0))

    def _render_ui(self):
        # Time Charges
        charges_text = self.font_ui.render(f"TIME CHARGES: {self.time_charges}", True, self.COLOR_UI_TEXT)
        self.screen.blit(charges_text, (10, 10))

        # Distance to Core Bar
        dist_bar_width = 200
        dist_bar_height = 20
        dist_bar_x = self.SCREEN_WIDTH - dist_bar_width - 10
        dist_bar_y = 10
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (dist_bar_x, dist_bar_y, dist_bar_width, dist_bar_height))
        
        current_dist = self._get_distance_to_core(self.current_section_id)
        fill_ratio = 1.0 - np.clip(current_dist / (self.max_dist_to_core + 1e-6), 0, 1)
        fill_width = dist_bar_width * fill_ratio
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FG, (dist_bar_x, dist_bar_y, fill_width, dist_bar_height))
        
        dist_text = self.font_ui.render("PROXIMITY TO CORE", True, self.COLOR_UI_TEXT)
        text_rect = dist_text.get_rect(center=(dist_bar_x + dist_bar_width/2, dist_bar_y + dist_bar_height/2))
        self.screen.blit(dist_text, text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "CORE REACHED" if self.current_section_id == self.core_id else "TIMED OUT"
            title_text = self.font_title.render(msg, True, self.COLOR_UI_TEXT)
            title_rect = title_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(title_text, title_rect)
            
            score_text = self.font_ui.render(f"Final Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
            score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20))
            self.screen.blit(score_text, score_rect)
            
    def _draw_gradient_background(self):
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_START[0] * (1 - ratio) + self.COLOR_BG_END[0] * ratio),
                int(self.COLOR_BG_START[1] * (1 - ratio) + self.COLOR_BG_END[1] * ratio),
                int(self.COLOR_BG_START[2] * (1 - ratio) + self.COLOR_BG_END[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_glowing_circle(self, pos, radius, color, glow_factor):
        pos_int = (int(pos[0]), int(pos[1]))
        # Draw multiple semi-transparent circles for the glow
        if glow_factor > 0:
            for i in range(4):
                glow_radius = int(radius + (i * 3 + 2) * glow_factor)
                alpha = int(80 * (1 - i / 4) * glow_factor)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], glow_radius, (*color, alpha))
        
        # Draw the main circle
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, color)

    def _create_particles(self, pos, count, color, min_speed, max_speed, target_pos=None):
        for _ in range(count):
            if target_pos: # particle stream
                angle = math.atan2(target_pos[1] - pos[1], target_pos[0] - pos[0])
                angle += self.np_random.uniform(-0.5, 0.5)
            else: # particle explosion
                angle = self.np_random.uniform(0, 2 * math.pi)
            
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['lifespan'] / 40))
                size = int(3 * (p['lifespan'] / 40))
                if size > 0:
                    try:
                        # Use SRCALPHA for particle surface to handle alpha correctly
                        particle_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                        pygame.draw.circle(particle_surf, (*p['color'], alpha), (size, size), size)
                        self.screen.blit(particle_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size), special_flags=pygame.BLEND_RGBA_ADD)
                    except (pygame.error, ValueError):
                        # Skip drawing if size is invalid
                        pass

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=random.randint(0, 10000))
    
    # --- Pygame setup for human play ---
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fractal Core")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Use a dictionary to track key presses for rising-edge detection
    key_pressed = {
        pygame.K_UP: False, pygame.K_w: False,
        pygame.K_DOWN: False, pygame.K_s: False,
        pygame.K_LEFT: False, pygame.K_a: False,
        pygame.K_RIGHT: False, pygame.K_d: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False, pygame.K_RSHIFT: False
    }
    
    while running:
        # --- Action mapping for human input ---
        movement = 0 # none
        space_held = 0
        shift_held = 0

        # Poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_pressed:
                    key_pressed[event.key] = True
                if event.key == pygame.K_r: # Reset on 'R' key
                    print("--- RESETTING ENVIRONMENT ---")
                    obs, info = env.reset(seed=random.randint(0, 10000))
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False
            if event.type == pygame.KEYUP:
                 if event.key in key_pressed:
                    key_pressed[event.key] = False

        # Map pressed keys to actions
        if key_pressed[pygame.K_UP] or key_pressed[pygame.K_w]: movement = 1
        elif key_pressed[pygame.K_DOWN] or key_pressed[pygame.K_s]: movement = 2
        elif key_pressed[pygame.K_LEFT] or key_pressed[pygame.K_a]: movement = 3
        elif key_pressed[pygame.K_RIGHT] or key_pressed[pygame.K_d]: movement = 4
        
        if key_pressed[pygame.K_SPACE]: space_held = 1
        if key_pressed[pygame.K_LSHIFT] or key_pressed[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # The game over screen will be displayed, wait for R to reset
        
        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()