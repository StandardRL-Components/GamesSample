import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:17:58.518091
# Source Brief: brief_00922.md
# Brief Index: 922
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gene Glitch: A Gymnasium environment where the agent must repair mutated gene sequences.

    The agent controls a cursor to place corrective enzymes. Time can be stopped to
    strategically aim. Placing the correct enzyme next to a matching mutated gene
    triggers a chain reaction, repairing all connected genes of the same type. The goal
    is to repair the entire sequence before running out of enzymes.

    Visuals are a key focus, with glowing particles, smooth animations, and a
    stylized microscopic aesthetic.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Repair mutated gene sequences by placing corrective enzymes. "
        "Strategically stop time to aim, and trigger chain reactions to clear the board before you run out of enzymes."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press Space to stop time for precise aiming, "
        "then Space again to place an enzyme. Press Shift to cycle through enzyme types."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Assumed FPS for visual effect timing
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.GENE_COLORS = [
            (255, 87, 87),   # Red
            (87, 255, 87),   # Green
            (87, 87, 255),   # Blue
            (255, 255, 87),  # Yellow
            (170, 87, 255),  # Purple
        ]
        self.COLOR_REPAIRED = (100, 100, 110)
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_BACKBONE = (40, 50, 75)
        self.COLOR_CURSOR_SELECT = (255, 255, 255, 50)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("Consolas", 18)
            self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_small = pygame.font.SysFont(None, 24)
            self.font_large = pygame.font.SysFont(None, 60)

        # --- State Variables ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.level = None
        self.time_stopped = None
        self.genes = None
        self.enzyme_counts = None
        self.selected_enzyme_type = None
        self.cursor_pos = None
        self.last_space_held = None
        self.last_shift_held = None
        self.particles = None
        self.bg_particles = None
        self.last_reward_info = None

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.time_stopped = True
        self.cursor_pos = np.array([self.WIDTH / 2.0, self.HEIGHT / 2.0])
        self.last_space_held = True # Prevent action on first frame
        self.last_shift_held = True # Prevent action on first frame
        self.particles = []
        self.last_reward_info = {'value': 0, 'timer': 0}

        if not hasattr(self, 'bg_particles') or self.bg_particles is None:
            self._init_bg_particles()

        self._setup_level()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        step_reward = 0

        if self.time_stopped:
            self._handle_stopped_time_actions(movement, space_pressed, shift_pressed)
            if space_pressed:
                # sound: place_enzyme
                step_reward += self._place_enzyme()
                self.time_stopped = False
        elif space_pressed:
            # sound: time_stop
            self.time_stopped = True

        self._update_game_state()
        self.steps += 1
        
        if step_reward != 0:
            self.last_reward_info = {'value': step_reward, 'timer': 60}

        self.score += step_reward

        terminated, terminal_reward = self._check_termination()
        if terminated:
            self.game_over = True
            if terminal_reward != 0:
                 self.last_reward_info = {'value': terminal_reward, 'timer': 120}

        total_reward = step_reward + terminal_reward
        self.score += terminal_reward

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), total_reward, terminated, truncated, self._get_info()

    def _handle_stopped_time_actions(self, movement, space_pressed, shift_pressed):
        move_speed = 15.0
        if movement == 1: self.cursor_pos[1] -= move_speed
        elif movement == 2: self.cursor_pos[1] += move_speed
        elif movement == 3: self.cursor_pos[0] -= move_speed
        elif movement == 4: self.cursor_pos[0] += move_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        if shift_pressed:
            # sound: UI_cycle
            num_types = len(self.GENE_COLORS)
            self.selected_enzyme_type = (self.selected_enzyme_type + 1) % num_types

    def _place_enzyme(self):
        if self.enzyme_counts[self.selected_enzyme_type] <= 0:
            # sound: error
            return 0

        closest_gene, min_dist_sq = self._find_closest_gene(self.cursor_pos)

        if closest_gene and min_dist_sq < (30**2) and \
           closest_gene['state'] == 'mutated' and \
           closest_gene['color_idx'] == self.selected_enzyme_type:
            
            self.enzyme_counts[self.selected_enzyme_type] -= 1
            # sound: chain_reaction_start
            return self._trigger_chain_reaction(closest_gene)
        else:
            # sound: fizzle
            self.enzyme_counts[self.selected_enzyme_type] -= 1
            return 0

    def _trigger_chain_reaction(self, start_gene):
        repaired_count = 0
        q = [start_gene]
        visited = {id(start_gene)}
        target_color_idx = start_gene['color_idx']

        while q:
            current_gene = q.pop(0)
            if current_gene['state'] == 'mutated' and current_gene['color_idx'] == target_color_idx:
                current_gene['state'] = 'repaired'
                repaired_count += 1
                # sound: gene_repaired
                self._add_particle_effect(current_gene['pos'], self.GENE_COLORS[target_color_idx])

                for neighbor in self.genes:
                    if id(neighbor) not in visited:
                        dist_sq = np.sum((current_gene['pos'] - neighbor['pos'])**2)
                        if dist_sq < (60**2): # Adjacency distance
                            visited.add(id(neighbor))
                            q.append(neighbor)
        
        return float(repaired_count)

    def _check_termination(self):
        mutated_remaining = any(g['state'] == 'mutated' for g in self.genes)

        if not mutated_remaining:
            # sound: level_complete
            return True, 50.0

        enzymes_remaining = sum(self.enzyme_counts) > 0
        if not enzymes_remaining and mutated_remaining:
            # sound: failure
            return True, -100.0
        
        if self.steps >= self.MAX_STEPS:
            return True, 0.0

        return False, 0.0

    def _update_game_state(self):
        for p in self.bg_particles:
            p['pos'] += p['vel']
            if p['pos'][0] < 0 or p['pos'][0] > self.WIDTH: p['vel'][0] *= -1
            if p['pos'][1] < 0 or p['pos'][1] > self.HEIGHT: p['vel'][1] *= -1

        for p in self.particles[:]:
            p['age'] += 1
            if p['age'] > p['lifespan']:
                self.particles.remove(p)
                continue
            p['pos'] += p['vel']
            p['radius'] += p['growth']

        for gene in self.genes:
            gene['pulse_phase'] = (gene['pulse_phase'] + 0.1) % (2 * math.pi)
            if gene['state'] == 'mutated':
                gene['radius'] = 8 + 2 * math.sin(gene['pulse_phase'])
            else:
                gene['radius'] = 6 + 0.5 * math.sin(gene['pulse_phase'])
        
        if self.last_reward_info['timer'] > 0:
            self.last_reward_info['timer'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for p in self.bg_particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), int(p['radius']))

        if len(self.genes) > 1:
            points = [g['pos'].astype(int) for g in sorted(self.genes, key=lambda g: g['pos'][0])]
            pygame.draw.lines(self.screen, self.COLOR_BACKBONE, False, points, 2)

        for gene in self.genes:
            pos = gene['pos'].astype(int)
            radius = int(gene['radius'])
            if gene['state'] == 'mutated':
                color = self.GENE_COLORS[gene['color_idx']]
                self._draw_glow_circle(pos, radius, color)
            else:
                color = self.COLOR_REPAIRED
                pygame.draw.circle(self.screen, color, pos, radius)
                pygame.draw.circle(self.screen, self.COLOR_BG, pos, radius - 2)

        for p in self.particles:
            if p['type'] == 'shockwave':
                alpha = max(0, 255 * (1 - p['age'] / p['lifespan']))
                self._draw_expanding_circle(p['pos'].astype(int), int(p['radius']), p['color'], alpha)

        if self.time_stopped:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            self.screen.blit(overlay, (0, 0))
            self._render_cursor()

    def _render_cursor(self):
        cursor_pos = self.cursor_pos.astype(int)
        cursor_color = self.GENE_COLORS[self.selected_enzyme_type]
        radius = 15
        angle = (pygame.time.get_ticks() / 500) * math.pi
        
        for i in range(2):
            offset = i * math.pi
            start_angle = angle + offset
            end_angle = start_angle + math.pi * 0.8
            pygame.draw.arc(self.screen, cursor_color, (cursor_pos[0]-radius, cursor_pos[1]-radius, radius*2, radius*2), start_angle, end_angle, 2)
        
        pygame.draw.circle(self.screen, cursor_color, cursor_pos, 4, 1)

    def _render_ui(self):
        level_surf = self.font_small.render(f"Level: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_surf, (10, 10))

        score_surf = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))

        if self.last_reward_info['timer'] > 0:
            val = self.last_reward_info['value']
            color = (150, 255, 150) if val > 0 else (255, 150, 150)
            text = f"+{int(val)}" if val > 0 else str(int(val))
            reward_surf = self.font_small.render(text, True, color)
            self.screen.blit(reward_surf, (self.WIDTH - reward_surf.get_width() - 10, 30))

        self._render_enzyme_inventory()

        if self.game_over:
            self._render_game_over_screen()

    def _render_enzyme_inventory(self):
        icon_size = 12
        spacing = 45
        num_types = len(self.GENE_COLORS)
        start_x = self.WIDTH / 2 - (num_types * spacing - (spacing - icon_size*2)) / 2
        
        for i, count in enumerate(self.enzyme_counts):
            x = int(start_x + i * spacing)
            y = self.HEIGHT - 25
            color = self.GENE_COLORS[i]
            
            if self.time_stopped and i == self.selected_enzyme_type:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR_SELECT, (x - 20, y - 15, 40, 30), border_radius=5)
                pygame.draw.circle(self.screen, color, (x, y), icon_size + 2)
            else:
                pygame.draw.circle(self.screen, color, (x, y), icon_size)

            count_surf = self.font_small.render(str(count), True, self.COLOR_UI_TEXT)
            self.screen.blit(count_surf, (x + icon_size + 2, y - count_surf.get_height() / 2))

    def _render_game_over_screen(self):
        mutated_remaining = any(g['state'] == 'mutated' for g in self.genes)
        text, color = ("LEVEL COMPLETE", (180, 255, 180)) if not mutated_remaining else ("FAILURE", (255, 180, 180))
        
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level, "time_stopped": self.time_stopped}

    def _init_bg_particles(self):
        self.bg_particles = []
        for _ in range(100):
            self.bg_particles.append({
                'pos': self.np_random.uniform(0, [self.WIDTH, self.HEIGHT], (2,)).astype(float),
                'vel': self.np_random.uniform(-0.2, 0.2, (2,)).astype(float),
                'radius': self.np_random.uniform(0.5, 1.5),
                'color': (30, 40, 60)
            })

    def _setup_level(self):
        self.genes = []
        num_mutated = 5 + (self.level - 1) * 2
        num_enzymes_per_type = 5 + (self.level - 1) // 3
        num_types = len(self.GENE_COLORS)

        self.enzyme_counts = [num_enzymes_per_type] * num_types
        self.selected_enzyme_type = 0

        path_points = []
        y_center = self.HEIGHT / 2
        amplitude = self.HEIGHT / 3.5
        frequency = self.np_random.uniform(2.5, 3.5)
        num_segments = num_mutated + self.np_random.integers(3, 7)
        for i in range(num_segments):
            x = self.WIDTH * (i + 1.5) / (num_segments + 2)
            y_wave = math.sin(i / num_segments * frequency * 2 * math.pi + self.np_random.uniform(-0.2, 0.2))
            y = y_center + amplitude * y_wave * math.sin(i / (num_segments-1) * math.pi) # Bow shape
            path_points.append(np.array([x, y]))

        placed_indices = self.np_random.choice(range(len(path_points)), num_mutated, replace=False)
        for i in placed_indices:
            self.genes.append({
                'pos': path_points[i],
                'color_idx': self.np_random.integers(0, num_types),
                'state': 'mutated',
                'radius': 8.0,
                'pulse_phase': self.np_random.uniform(0, 2 * math.pi)
            })

    def _find_closest_gene(self, position):
        closest_gene = None
        min_dist_sq = float('inf')
        if not self.genes: return None, min_dist_sq
        
        gene_positions = np.array([g['pos'] for g in self.genes])
        diffs = gene_positions - position
        dist_sqs = np.sum(diffs**2, axis=1)
        
        min_idx = np.argmin(dist_sqs)
        return self.genes[min_idx], dist_sqs[min_idx]

    def _add_particle_effect(self, pos, color):
        self.particles.append({
            'pos': pos.copy(), 'vel': np.array([0.0, 0.0]),
            'radius': 5.0, 'growth': 1.0, 'age': 0, 'lifespan': 20,
            'color': color, 'type': 'shockwave'
        })

    def _draw_glow_circle(self, pos, radius, color):
        for i in range(4):
            alpha = 150 / (i + 1)
            current_radius = radius + i * 2
            s = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (current_radius, current_radius), current_radius)
            self.screen.blit(s, (pos[0] - current_radius, pos[1] - current_radius))
        pygame.draw.circle(self.screen, color, pos, int(radius))

    def _draw_expanding_circle(self, pos, radius, color, alpha):
        if radius <= 0: return
        try:
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*color, int(alpha)))
        except (ValueError, TypeError): # Handle potential errors with large radius or invalid color
            pass

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a graphical display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Setup ---
    # Map keys to MultiDiscrete actions
    # [Movement, Space, Shift]
    key_map = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
    }
    
    # --- Pygame Window for Manual Play ---
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gene Glitch - Manual Test")
    clock = pygame.time.Clock()
    
    while not done:
        # --- Action Generation ---
        action = [0, 0, 0] # Default action: no-op
        keys = pygame.key.get_pressed()

        for key, act in key_map.items():
            if keys[key]:
                action = act
                break # Prioritize first key found
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is a numpy array, convert it back to a Pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        clock.tick(env.FPS)
        
    print(f"Game Over. Final Score: {info['score']}")
    pygame.quit()