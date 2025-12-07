import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:42:13.965696
# Source Brief: brief_02319.md
# Brief Index: 2319
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Play as a spreading cancer, colonizing tissues and fighting the immune system. "
        "Create portals to metastasize and activate genes to enhance your growth and invasion."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor or select genes. "
        "Press space to place portals or activate genes. Press shift to switch modes."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Game Constants
        self.screen_width, self.screen_height = 640, 400
        self.ui_width = 180
        self.game_width = self.screen_width - self.ui_width
        self.max_steps = 1000
        self.cursor_speed = 10
        self.initial_cell_population = 100

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans", 14)
        self.font_medium = pygame.font.SysFont("sans", 18, bold=True)
        self.font_large = pygame.font.SysFont("sans", 24, bold=True)

        # Colors
        self.COLOR_BG = (10, 10, 26)
        self.COLOR_UI_BG = (20, 20, 40)
        self.COLOR_UI_BORDER = (40, 40, 60)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_DIM = (150, 150, 150)
        self.COLOR_TEXT_SUCCESS = (100, 255, 100)
        self.COLOR_TEXT_DANGER = (255, 100, 100)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_HEALTHY_TISSUE = (25, 61, 44)
        self.COLOR_CANCER_TISSUE_BG = (80, 20, 30)
        self.COLOR_CANCER_CELLS = (255, 50, 80)
        self.COLOR_IMMUNE_CELLS = (50, 150, 255)
        self.COLOR_PORTAL = (200, 50, 255)
        self.COLOR_PENDING_PORTAL = (255, 255, 255)

        # Initialize state variables
        # These will be properly set in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tissues = []
        self.immune_cells = []
        self.portals = []
        self.particles = []
        self.genes = []
        self.cursor_pos = [0, 0]
        self.selection_mode = 'portal'
        self.selected_gene_idx = 0
        self.pending_portal_tissue_idx = None
        self.last_space_held = False
        self.last_shift_held = False
        self.immune_strength_multiplier = 1.0
        self.status_message = ""
        self.status_message_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Game state
        self._initialize_tissues()
        self._initialize_genes()
        self.immune_cells = []
        self.portals = []
        self.particles = []
        self.immune_strength_multiplier = 1.0

        # Player state
        self.cursor_pos = [self.game_width // 2, self.screen_height // 2]
        self.selection_mode = 'portal'
        self.selected_gene_idx = 0
        self.pending_portal_tissue_idx = None
        self.last_space_held = False
        self.last_shift_held = False
        
        # Initial immune cells
        for _ in range(2):
            self._spawn_immune_cell()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        self.status_message_timer = max(0, self.status_message_timer - 1)

        # 1. Handle Player Input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        if shift_pressed:
            self.selection_mode = 'gene' if self.selection_mode == 'portal' else 'portal'
            self.status_message = f"Mode: {self.selection_mode.upper()}"
            self.status_message_timer = 60

        self._handle_movement(movement)

        if space_pressed:
            action_reward, action_message = self._handle_action()
            reward += action_reward
            if action_message:
                self.status_message = action_message
                self.status_message_timer = 90
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # 2. Update Game Logic
        update_rewards = self._update_game_state()
        reward += update_rewards

        # 3. Check for Termination
        terminated = False
        truncated = False
        total_population = sum(t['population'] for t in self.tissues)
        colonized_tissues = sum(1 for t in self.tissues if t['colonized'])

        if total_population <= 0:
            terminated = True
            self.game_over = True
            reward -= 100
            self.status_message = "FAILURE: Cancer Eradicated"
            self.status_message_timer = 1000
        elif colonized_tissues == len(self.tissues):
            terminated = True
            self.game_over = True
            reward += 100
            self.status_message = "VICTORY: Full Metastasis"
            self.status_message_timer = 1000
        elif self.steps >= self.max_steps:
            truncated = True
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

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
            "total_population": sum(t['population'] for t in self.tissues),
            "colonized_tissues": sum(1 for t in self.tissues if t['colonized']),
        }

    # --- Game Logic Update ---
    def _update_game_state(self):
        reward = 0
        
        # Cancer Growth & Spread
        for i, tissue in enumerate(self.tissues):
            if tissue['colonized']:
                growth_rate = 0.01 * (1 + sum(g['value'] for g in self.genes if g['active'] and g['effect'] == 'growth'))
                tissue['population'] *= (1 + growth_rate)
        
        for p_start, p_end in self.portals:
            t_start, t_end = self.tissues[p_start], self.tissues[p_end]
            if t_start['colonized'] and t_start['population'] > 10:
                resistance_mod = sum(g['value'] for g in self.genes if g['active'] and g['effect'] == 'invasion')
                effective_resistance = max(0.1, t_end['resistance'] - resistance_mod)
                spread_pressure = t_start['population'] / (effective_resistance * 5000)
                
                if self.np_random.random() < spread_pressure:
                    spread_amount = min(t_start['population'] * 0.05, 5)
                    if spread_amount > 0:
                        t_start['population'] -= spread_amount
                        t_end['population'] += spread_amount
                        if not t_end['colonized']:
                            t_end['colonized'] = True
                            reward += 2 # Invaded new tissue
                            # Sfx: tissue_invaded_chime

        # Immune Cell Action
        for immune_cell in self.immune_cells:
            if immune_cell['target_idx'] is None or not self.tissues[immune_cell['target_idx']]['colonized']:
                colonized_indices = [i for i, t in enumerate(self.tissues) if t['colonized']]
                if colonized_indices:
                    immune_cell['target_idx'] = self.np_random.choice(colonized_indices)

            if immune_cell['target_idx'] is not None:
                target_pos = self.tissues[immune_cell['target_idx']]['rect'].center
                direction = (target_pos[0] - immune_cell['pos'][0], target_pos[1] - immune_cell['pos'][1])
                dist = math.hypot(*direction)
                if dist > 1:
                    immune_cell['pos'][0] += direction[0] / dist * 2
                    immune_cell['pos'][1] += direction[1] / dist * 2
                
                if self.tissues[immune_cell['target_idx']]['rect'].collidepoint(immune_cell['pos']):
                    tissue = self.tissues[immune_cell['target_idx']]
                    damage = immune_cell['strength'] * self.immune_strength_multiplier
                    cells_lost = min(tissue['population'], damage)
                    tissue['population'] -= cells_lost
                    reward -= cells_lost * 0.1
                    if self.np_random.random() < 0.1:
                        self._spawn_particles(immune_cell['pos'], self.COLOR_IMMUNE_CELLS, 3, 5) # Sfx: immune_attack_zap

        # Progression
        if self.steps > 0:
            if self.steps % 20 == 0:
                self.immune_strength_multiplier *= 1.1
            if self.steps % 30 == 0:
                unlocked_any = False
                for gene in self.genes:
                    if not gene['unlocked']:
                        gene['unlocked'] = True
                        reward += 5 # New gene unlocked
                        self.status_message = f"GENE UNLOCKED: {gene['name']}"
                        self.status_message_timer = 120
                        unlocked_any = True
                        break
            if self.steps % 50 == 0 and len(self.tissues) < 8:
                self._add_new_tissue()

        return reward

    # --- Action Handling ---
    def _handle_movement(self, movement):
        if self.selection_mode == 'portal':
            if movement == 1: self.cursor_pos[1] -= self.cursor_speed
            elif movement == 2: self.cursor_pos[1] += self.cursor_speed
            elif movement == 3: self.cursor_pos[0] -= self.cursor_speed
            elif movement == 4: self.cursor_pos[0] += self.cursor_speed
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.game_width - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.screen_height - 1)
        elif self.selection_mode == 'gene':
            num_genes = len(self.genes)
            if num_genes > 0:
                if movement == 1: self.selected_gene_idx = (self.selected_gene_idx - 1 + num_genes) % num_genes
                elif movement == 2: self.selected_gene_idx = (self.selected_gene_idx + 1) % num_genes
    
    def _handle_action(self):
        if self.selection_mode == 'portal':
            return self._handle_portal_placement()
        elif self.selection_mode == 'gene':
            return self._handle_gene_activation()
        return 0, ""

    def _handle_portal_placement(self):
        hovered_tissue_idx = None
        for i, tissue in enumerate(self.tissues):
            if tissue['rect'].collidepoint(self.cursor_pos):
                hovered_tissue_idx = i
                break
        
        if hovered_tissue_idx is None:
            return 0, "Invalid portal location"

        if self.pending_portal_tissue_idx is None:
            if not self.tissues[hovered_tissue_idx]['colonized']:
                return 0, "Must start portal in colonized tissue"
            self.pending_portal_tissue_idx = hovered_tissue_idx
            # Sfx: portal_start_ping
            return 0, "Portal start set. Select destination."
        else:
            if self.pending_portal_tissue_idx == hovered_tissue_idx:
                self.pending_portal_tissue_idx = None
                return 0, "Portal placement cancelled"
            
            portal_tuple = tuple(sorted((self.pending_portal_tissue_idx, hovered_tissue_idx)))
            if portal_tuple in self.portals:
                return 0, "Portal already exists"
                
            self.portals.append(portal_tuple)
            start_pos = self.tissues[self.pending_portal_tissue_idx]['rect'].center
            end_pos = self.tissues[hovered_tissue_idx]['rect'].center
            self._spawn_particles(start_pos, self.COLOR_PORTAL, 10, 8)
            self._spawn_particles(end_pos, self.COLOR_PORTAL, 10, 8)
            self.pending_portal_tissue_idx = None
            # Sfx: portal_connect_whoosh
            return 0.5, "Portal created!"

    def _handle_gene_activation(self):
        if not self.genes: return 0, "No genes available"
        gene = self.genes[self.selected_gene_idx]
        if not gene['unlocked']:
            return 0, "Gene not unlocked"
        
        gene['active'] = not gene['active']
        # Sfx: gene_activate_click
        status = "ACTIVATED" if gene['active'] else "DEACTIVATED"
        return 0.1, f"{gene['name']} {status}"

    # --- Rendering ---
    def _render_game(self):
        # Draw Tissues
        for tissue in self.tissues:
            color = self.COLOR_CANCER_TISSUE_BG if tissue['colonized'] else self.COLOR_HEALTHY_TISSUE
            pygame.draw.rect(self.screen, color, tissue['rect'])
            pygame.gfxdraw.rectangle(self.screen, tissue['rect'], (*color, 150))
            if tissue['colonized']:
                pop_ratio = min(1, tissue['population'] / 1000)
                pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 0.2 + 0.8
                w = int(tissue['rect'].width * pop_ratio * pulse)
                h = int(tissue['rect'].height * pop_ratio * pulse)
                if w > 0 and h > 0:
                    blob_rect = pygame.Rect(0, 0, w, h)
                    blob_rect.center = tissue['rect'].center
                    self._draw_glowing_circle(self.screen, self.COLOR_CANCER_CELLS, blob_rect.center, blob_rect.width // 2, 10)

        # Draw Portals
        for start_idx, end_idx in self.portals:
            start_pos = self.tissues[start_idx]['rect'].center
            end_pos = self.tissues[end_idx]['rect'].center
            self._draw_glowing_line(self.screen, self.COLOR_PORTAL, start_pos, end_pos, 3)
            self._draw_glowing_circle(self.screen, self.COLOR_PORTAL, start_pos, 8, 5)
            self._draw_glowing_circle(self.screen, self.COLOR_PORTAL, end_pos, 8, 5)

        # Draw Pending Portal
        if self.pending_portal_tissue_idx is not None:
            start_pos = self.tissues[self.pending_portal_tissue_idx]['rect'].center
            self._draw_glowing_line(self.screen, self.COLOR_PENDING_PORTAL, start_pos, self.cursor_pos, 1)
            self._draw_glowing_circle(self.screen, self.COLOR_PENDING_PORTAL, start_pos, 10, 8)

        # Draw Immune Cells
        for cell in self.immune_cells:
            self._draw_glowing_circle(self.screen, self.COLOR_IMMUNE_CELLS, cell['pos'], 6, 4)
            pygame.gfxdraw.filled_circle(self.screen, int(cell['pos'][0]), int(cell['pos'][1]), 6, self.COLOR_IMMUNE_CELLS)

        # Draw Particles
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            alpha = p['life'] / p['max_life']
            color = (*p['color'], int(255 * alpha))
            radius = int(p['radius'] * alpha)
            if radius > 0:
                self._draw_glowing_circle(self.screen, color, p['pos'], radius, 0)
        
        # Draw Cursor
        if self.selection_mode == 'portal':
            self._draw_glowing_circle(self.screen, self.COLOR_CURSOR, self.cursor_pos, 10, 5)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor_pos[0]-8, self.cursor_pos[1]), (self.cursor_pos[0]+8, self.cursor_pos[1]), 1)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor_pos[0], self.cursor_pos[1]-8), (self.cursor_pos[0], self.cursor_pos[1]+8), 1)

    def _render_ui(self):
        ui_rect = pygame.Rect(self.game_width, 0, self.ui_width, self.screen_height)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_UI_BORDER, (self.game_width, 0), (self.game_width, self.screen_height), 2)
        
        y = 15
        self._draw_text("STATUS", self.game_width + 15, y, font=self.font_large, color=self.COLOR_TEXT)
        y += 35
        self._draw_text(f"Score: {int(self.score)}", self.game_width + 15, y, font=self.font_medium)
        y += 25
        self._draw_text(f"Step: {self.steps}/{self.max_steps}", self.game_width + 15, y, font=self.font_medium)
        y += 25
        total_pop = sum(t['population'] for t in self.tissues)
        self._draw_text(f"Population: {int(total_pop)}", self.game_width + 15, y, font=self.font_medium)
        y += 40

        self._draw_text("GENES/TOXINS", self.game_width + 15, y, font=self.font_large, color=self.COLOR_TEXT)
        y += 35
        
        for i, gene in enumerate(self.genes):
            if i == self.selected_gene_idx and self.selection_mode == 'gene':
                sel_rect = pygame.Rect(self.game_width + 5, y - 5, self.ui_width - 10, 25)
                pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, sel_rect, 0, 5)
            
            if not gene['unlocked']:
                self._draw_text(f"[{i+1}] ???????", self.game_width + 15, y, color=self.COLOR_TEXT_DIM)
            else:
                color = self.COLOR_TEXT_SUCCESS if gene['active'] else self.COLOR_TEXT
                self._draw_text(f"[{i+1}] {gene['name']}", self.game_width + 15, y, color=color)
            y += 25
        
        # Status Message
        if self.status_message and self.status_message_timer > 0:
            alpha = min(255, self.status_message_timer * 5)
            text_surf = self.font_medium.render(self.status_message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.game_width / 2, self.screen_height - 30))
            bg_rect = text_rect.inflate(20, 10)
            
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((*self.COLOR_UI_BG, alpha * 0.7))
            pygame.draw.rect(s, (*self.COLOR_UI_BORDER, alpha), s.get_rect(), 2, 5)
            text_surf.set_alpha(alpha)

            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(text_surf, text_rect.topleft)

    # --- Initialization ---
    def _initialize_tissues(self):
        self.tissues = []
        # Create a grid of potential tissue locations
        cols, rows = 3, 2
        padding = 20
        cell_w = (self.game_width - (cols + 1) * padding) / cols
        cell_h = (self.screen_height - (rows + 1) * padding) / rows

        for r in range(rows):
            for c in range(cols):
                x = padding + c * (cell_w + padding)
                y = padding + r * (cell_h + padding)
                self.tissues.append({
                    "rect": pygame.Rect(x, y, cell_w, cell_h),
                    "resistance": 1.0 + (r + c) * 0.2,
                    "population": 0,
                    "colonized": False,
                })

        # Colonize the first tissue
        start_idx = 0
        self.tissues[start_idx]['population'] = self.initial_cell_population
        self.tissues[start_idx]['colonized'] = True

    def _initialize_genes(self):
        self.genes = [
            {'name': 'Growth+', 'effect': 'growth', 'value': 0.5, 'unlocked': True, 'active': False},
            {'name': 'Invasin', 'effect': 'invasion', 'value': 0.5, 'unlocked': False, 'active': False},
            {'name': 'Hyper-Growth', 'effect': 'growth', 'value': 1.0, 'unlocked': False, 'active': False},
            {'name': 'Armor', 'effect': 'invasion', 'value': -0.3, 'unlocked': False, 'active': False}, # This is a negative invasion, i.e. defense
        ]

    def _add_new_tissue(self):
        # Placeholder for adding more complex tissues
        pass

    def _spawn_immune_cell(self):
        x = self.np_random.choice([0, self.game_width])
        y = self.np_random.uniform(0, self.screen_height)
        self.immune_cells.append({
            "pos": [x, y],
            "strength": self.np_random.uniform(2, 5),
            "target_idx": None,
        })
    
    # --- Drawing Helpers ---
    def _draw_text(self, text, x, y, font=None, color=None):
        if font is None: font = self.font_small
        if color is None: color = self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def _spawn_particles(self, pos, color, count, radius):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color,
                'radius': radius
            })

    def _draw_glowing_circle(self, surface, color, pos, radius, glow_size):
        if radius <= 0: return
        pos = (int(pos[0]), int(pos[1]))
        
        # Create a temporary surface for the glow
        glow_radius = int(radius + glow_size)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        
        # Draw a filled circle on it with falloff
        for i in range(glow_radius, 0, -1):
            alpha = int(255 * (1 - i / glow_radius)**2)
            if len(color) == 4:
                final_alpha = int(alpha * (color[3] / 255.0))
                pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, i, (*color[:3], final_alpha))
            else:
                pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, i, (*color, alpha))
        
        surface.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Draw the main circle
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius), color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius), color)

    def _draw_glowing_line(self, surface, color, start_pos, end_pos, width):
        glow_width = int(width * 3)
        
        # Draw wider, transparent lines for the glow
        for i in range(glow_width, 0, -2):
            alpha = int(80 * (1 - i / glow_width))
            pygame.draw.line(surface, (*color[:3], alpha), start_pos, end_pos, i)

        # Draw the main line
        pygame.draw.line(surface, color, start_pos, end_pos, width)


    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It is not used by the evaluation environment.
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Use a display for manual playing
    pygame.display.set_caption("Cancer Metastasis Gym Env")
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()

    while running:
        if terminated:
            # After an episode ends, wait for a key press to reset
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    obs, info = env.reset()
                    terminated = False
        else:
            # Map keyboard keys to actions for manual control
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    env.close()