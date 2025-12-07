import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:34:12.030139
# Source Brief: brief_01063.md
# Brief Index: 1063
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
        "Manage an ancient aqueduct system. Adjust the channels to deliver water to growing settlements and keep them happy."
    )
    user_guide = (
        "Use ↑ and ↓ arrow keys to select an aqueduct. Use ← and → to shrink or grow the selected aqueduct to control water flow."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (40, 30, 20)
    COLOR_WATER = (100, 150, 255)
    COLOR_AQUEDUCT = (150, 140, 130)
    COLOR_AQUEDUCT_BORDER = (100, 90, 80)
    COLOR_SETTLEMENT_HAPPY = (50, 200, 50)
    COLOR_SETTLEMENT_SAD = (200, 50, 50)
    COLOR_TEXT = (240, 230, 220)
    COLOR_HIGHLIGHT = (255, 255, 0)
    COLOR_SOURCE = (150, 200, 255)

    # Game Parameters
    MAX_STEPS = 1000
    INITIAL_LEVEL = 1
    HAPPINESS_MAX = 100
    HAPPINESS_START = 50
    HAPPINESS_DECAY = 0.5
    WATER_PER_HAPPINESS = 0.1
    SETTLEMENT_CONSUMPTION = 1.0
    MIN_AQUEDUCT_SIZE = 5
    MAX_AQUEDUCT_SIZE = 25
    AQUEDUCT_RESIZE_STEP = 1
    PARTICLE_LIFETIME = 30
    PARTICLE_SPAWN_RATE = 2.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans", 18)
        self.font_large = pygame.font.SysFont("sans", 24)
        
        self.render_mode = render_mode
        self.level = self.INITIAL_LEVEL

        # Initialize state variables to be defined in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.settlements = []
        self.aqueducts = []
        self.water_levels = np.array([])
        self.downstream_map = {}
        self.selected_aqueduct_idx = 0
        self.particles = []
        self.source_pos = pygame.Vector2(0, 0)
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # validation is done by the test suite

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and 'level' in options:
            self.level = options['level']

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        self._generate_level()

        self.selected_aqueduct_idx = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        # --- Action Handling ---
        action_taken = True
        if movement == 1: # Up
            if self.aqueducts:
                self.selected_aqueduct_idx = (self.selected_aqueduct_idx - 1 + len(self.aqueducts)) % len(self.aqueducts)
        elif movement == 2: # Down
            if self.aqueducts:
                self.selected_aqueduct_idx = (self.selected_aqueduct_idx + 1) % len(self.aqueducts)
        elif movement == 3: # Left (Shrink)
            if self.aqueducts:
                aq = self.aqueducts[self.selected_aqueduct_idx]
                aq['size'] = max(self.MIN_AQUEDUCT_SIZE, aq['size'] - self.AQUEDUCT_RESIZE_STEP)
        elif movement == 4: # Right (Grow)
            if self.aqueducts:
                aq = self.aqueducts[self.selected_aqueduct_idx]
                aq['size'] = min(self.MAX_AQUEDUCT_SIZE, aq['size'] + self.AQUEDUCT_RESIZE_STEP)
        else: # No-op
            action_taken = False
        
        # In this game, time passes and settlements consume water even with no-op
        self.steps += 1

        # --- Game Logic Update ---
        old_happiness = np.array([s['happiness'] for s in self.settlements])
        self._update_water_flow()
        self._update_settlements()
        
        # --- Reward Calculation ---
        reward = self._calculate_reward(old_happiness)
        self.score += reward

        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if all(s['happiness'] >= self.HAPPINESS_MAX for s in self.settlements):
                 self.level += 1 # Progress to next level on next reset

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        self.settlements.clear()
        self.aqueducts.clear()
        self.downstream_map.clear()

        num_settlements = min(self.level + 1, 6)
        num_junctions = min(self.level + 2, 8)
        
        nodes = []
        self.source_pos = pygame.Vector2(self.np_random.integers(200, self.SCREEN_WIDTH - 200), self.np_random.integers(40, 60))
        nodes.append({'pos': self.source_pos, 'type': 'source'})

        # Generate settlements
        settlement_y = self.np_random.integers(self.SCREEN_HEIGHT - 80, self.SCREEN_HEIGHT - 40)
        settlement_xs = np.linspace(80, self.SCREEN_WIDTH - 80, num_settlements)
        for i in range(num_settlements):
            pos = pygame.Vector2(settlement_xs[i] + self.np_random.integers(-20, 20), settlement_y + self.np_random.integers(-10, 10))
            nodes.append({'pos': pos, 'type': 'settlement'})
            self.settlements.append({
                'id': i,
                'pos': pos,
                'happiness': self.HAPPINESS_START,
                'aq_idx': -1
            })

        # Generate junctions
        for _ in range(num_junctions):
            pos = pygame.Vector2(self.np_random.integers(50, self.SCREEN_WIDTH - 50), self.np_random.integers(100, self.SCREEN_HEIGHT - 100))
            nodes.append({'pos': pos, 'type': 'junction'})

        # Create aqueduct connections (DAG)
        nodes.sort(key=lambda n: n['pos'].y)
        
        for i in range(len(nodes) -1):
            start_node = nodes[i]
            # Connect to at least one node below
            possible_targets = [n for n in nodes[i+1:] if n['pos'].y > start_node['pos'].y + 10]
            if not possible_targets:
                continue
            
            # Guaranteed connection to create a path
            target_node = min(possible_targets, key=lambda n: n['pos'].distance_to(start_node['pos']))
            self._add_aqueduct(start_node['pos'], target_node['pos'])

            # Add extra random connections for complexity
            if self.np_random.random() < 0.3 and len(possible_targets) > 1:
                extra_target_candidates = [n for n in possible_targets if n != target_node]
                if extra_target_candidates:
                    extra_target = self.np_random.choice(extra_target_candidates)
                    self._add_aqueduct(start_node['pos'], extra_target['pos'])

        # Build downstream map and link settlements
        self.downstream_map = {i: [] for i in range(len(self.aqueducts))}
        for i, aq1 in enumerate(self.aqueducts):
            for j, aq2 in enumerate(self.aqueducts):
                if i != j and aq1['end_pos'] == aq2['start_pos']:
                    self.downstream_map[i].append(j)
        
        for s in self.settlements:
            for i, aq in enumerate(self.aqueducts):
                if aq['end_pos'] == s['pos'] and not self.downstream_map.get(i):
                    s['aq_idx'] = i
                    break
        
        self.water_levels = np.zeros(len(self.aqueducts), dtype=float)

    def _add_aqueduct(self, start_pos, end_pos):
        size = self.np_random.integers(self.MIN_AQUEDUCT_SIZE, self.MAX_AQUEDUCT_SIZE + 1)
        self.aqueducts.append({
            'start_pos': start_pos,
            'end_pos': end_pos,
            'size': size,
            'id': len(self.aqueducts)
        })

    def _update_water_flow(self):
        if not self.aqueducts:
            return

        # Source aqueducts are always full
        source_indices = [i for i, aq in enumerate(self.aqueducts) if aq['start_pos'] == self.source_pos]
        for idx in source_indices:
            self.water_levels[idx] = self.aqueducts[idx]['size']

        capacities = np.array([aq['size'] for aq in self.aqueducts])
        inflows = np.zeros_like(self.water_levels)

        # Iterate from top to bottom implicitly by aqueduct index order (due to generation)
        for i in range(len(self.aqueducts)):
            current_water = self.water_levels[i]
            if current_water <= 0:
                continue

            downstream_indices = self.downstream_map.get(i, [])
            if not downstream_indices:
                continue

            # Distribute water to downstream pipes
            num_downstream = len(downstream_indices)
            flow_per_pipe = current_water / num_downstream
            
            total_outflow = 0
            for j in downstream_indices:
                available_space = capacities[j] - self.water_levels[j] - inflows[j]
                actual_flow = min(flow_per_pipe, max(0, available_space))
                
                inflows[j] += actual_flow
                total_outflow += actual_flow
                
                # Spawn particles
                if actual_flow > 0.1 and self.np_random.random() < 0.5:
                    num_particles = int(actual_flow * self.PARTICLE_SPAWN_RATE)
                    for _ in range(num_particles):
                        self._spawn_particle(self.aqueducts[j])

            inflows[i] -= total_outflow
        
        self.water_levels += inflows
        self.water_levels = np.clip(self.water_levels, 0, capacities)

    def _spawn_particle(self, aqueduct):
        vec = aqueduct['end_pos'] - aqueduct['start_pos']
        pos = aqueduct['start_pos'] + vec * self.np_random.random()
        vel = vec.normalize() * (1 + self.np_random.random() * 2)
        self.particles.append({
            'pos': pos,
            'vel': vel,
            'life': self.PARTICLE_LIFETIME,
            'size': self.np_random.integers(1, 3)
        })

    def _update_settlements(self):
        for s in self.settlements:
            if s['aq_idx'] != -1 and self.water_levels[s['aq_idx']] > 0:
                water_consumed = min(self.water_levels[s['aq_idx']], self.SETTLEMENT_CONSUMPTION)
                self.water_levels[s['aq_idx']] -= water_consumed
                s['happiness'] += water_consumed / self.WATER_PER_HAPPINESS
            else:
                s['happiness'] -= self.HAPPINESS_DECAY
            s['happiness'] = max(0, min(self.HAPPINESS_MAX, s['happiness']))

    def _calculate_reward(self, old_happiness):
        reward = 0
        new_happiness = np.array([s['happiness'] for s in self.settlements])
        
        # Continuous reward for happiness change
        happiness_diff = new_happiness - old_happiness
        reward += np.sum(np.sign(happiness_diff))
        
        # Event-based rewards
        for i in range(len(self.settlements)):
            if new_happiness[i] >= self.HAPPINESS_MAX and old_happiness[i] < self.HAPPINESS_MAX:
                reward += 10  # Settlement fully supplied
            if new_happiness[i] <= 0 and old_happiness[i] > 0:
                reward -= 100 # Settlement revolted
        
        if all(h >= self.HAPPINESS_MAX for h in new_happiness):
            reward += 100 # Level complete
        
        return float(reward)

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if any(s['happiness'] <= 0 for s in self.settlements):
            return True
        if all(s['happiness'] >= self.HAPPINESS_MAX for s in self.settlements):
            return True
        return False

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
            "level": self.level,
            "settlements_happy": sum(1 for s in self.settlements if s['happiness'] >= self.HAPPINESS_MAX),
            "total_settlements": len(self.settlements)
        }

    def _render_game(self):
        self._render_aqueducts()
        self._render_particles()
        self._render_source()
        self._render_settlements()

    def _render_aqueducts(self):
        for i, aq in enumerate(self.aqueducts):
            is_selected = (i == self.selected_aqueduct_idx)
            
            # Draw glow for selected aqueduct
            if is_selected:
                for j in range(5, 0, -1):
                    glow_color = (*self.COLOR_HIGHLIGHT, 8 * j)
                    self._draw_thick_line_as_polygon(
                        self.screen, aq['start_pos'], aq['end_pos'], aq['size'] + j * 2, glow_color, use_gfx=True
                    )

            # Draw main aqueduct structure
            self._draw_thick_line_as_polygon(
                self.screen, aq['start_pos'], aq['end_pos'], aq['size'], self.COLOR_AQUEDUCT
            )
            self._draw_thick_line_as_polygon(
                self.screen, aq['start_pos'], aq['end_pos'], aq['size'], self.COLOR_AQUEDUCT_BORDER, 2
            )
            
            # Draw water level
            water_percentage = self.water_levels[i] / max(1, aq['size'])
            if water_percentage > 0:
                water_width = aq['size'] * water_percentage
                self._draw_thick_line_as_polygon(
                    self.screen, aq['start_pos'], aq['end_pos'], water_width, self.COLOR_WATER
                )

    def _render_particles(self):
        # Update and draw particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / self.PARTICLE_LIFETIME))
                color = (*self.COLOR_WATER, alpha)
                try:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), color)
                except OverflowError: # Catch rare errors with particle positions
                    if p in self.particles: self.particles.remove(p)


    def _render_source(self):
        pygame.gfxdraw.filled_circle(self.screen, int(self.source_pos.x), int(self.source_pos.y), 15, self.COLOR_SOURCE)
        pygame.gfxdraw.aacircle(self.screen, int(self.source_pos.x), int(self.source_pos.y), 15, self.COLOR_TEXT)

    def _render_settlements(self):
        for s in self.settlements:
            # Interpolate color based on happiness
            happy_ratio = s['happiness'] / self.HAPPINESS_MAX
            color = pygame.Color(self.COLOR_SETTLEMENT_SAD).lerp(self.COLOR_SETTLEMENT_HAPPY, happy_ratio)
            
            # Draw simple houses
            base_x, base_y = int(s['pos'].x), int(s['pos'].y)
            pygame.draw.rect(self.screen, color, (base_x - 10, base_y - 10, 20, 10))
            pygame.draw.polygon(self.screen, color, [(base_x - 12, base_y - 10), (base_x + 12, base_y - 10), (base_x, base_y - 20)])

            # Draw happiness bar
            bar_width = 40
            bar_height = 5
            bar_x = base_x - bar_width // 2
            bar_y = base_y + 5
            pygame.draw.rect(self.screen, self.COLOR_AQUEDUCT_BORDER, (bar_x, bar_y, bar_width, bar_height))
            fill_width = int(bar_width * happy_ratio)
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height))

    def _render_ui(self):
        # Score and Steps
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font_small.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 40))
        
        # Level
        level_text = self.font_large.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))

    @staticmethod
    def _draw_thick_line_as_polygon(surface, p1, p2, thickness, color, border_width=0, use_gfx=False):
        v = p2 - p1
        if v.length() == 0: return
        
        n = v.normalize().rotate(90) * (thickness / 2)
        
        points = [p1 + n, p2 + n, p2 - n, p1 - n]
        int_points = [(int(p.x), int(p.y)) for p in points]
        
        if use_gfx:
            pygame.gfxdraw.aapolygon(surface, int_points, color)
            pygame.gfxdraw.filled_polygon(surface, int_points, color)
        elif border_width > 0:
            pygame.draw.polygon(surface, color, int_points, border_width)
        else:
            pygame.draw.polygon(surface, color, int_points)

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    done = False
    
    # Pygame window for manual playing
    pygame.display.set_caption("Aqueduct Manager")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    action = [0, 0, 0] # [movement, space, shift]
    
    while not done:
        # Get observation from env and display it
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Reset action
        action = [0, 0, 0]
        
        # Handle user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                elif event.key == pygame.K_q: # Quit on 'q'
                    done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset() # Auto-reset for continuous play
            
        env.clock.tick(GameEnv.FPS)
        
    env.close()