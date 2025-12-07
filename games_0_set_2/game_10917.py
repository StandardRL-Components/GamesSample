import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:12:22.315657
# Source Brief: brief_00917.md
# Brief Index: 917
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
        "Synthesize a target DNA sequence by connecting complementary base pairs. "
        "Race against the clock in this molecular puzzle game."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to select/connect "
        "DNA segments. Hold shift to speed up time."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode

        # Visuals & Game Constants
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_GRID = (20, 40, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 255)
        self.DNA_COLORS = {
            'A': (255, 80, 80), 'T': (80, 255, 255),
            'C': (255, 255, 80), 'G': (80, 80, 255)
        }
        self.DNA_TYPES = list(self.DNA_COLORS.keys())
        self.DNA_COMPLEMENTS = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        
        try:
            self.font_main = pygame.font.SysFont("Consolas", 20)
            self.font_title = pygame.font.SysFont("Consolas", 28, bold=True)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 24)
            self.font_title = pygame.font.Font(None, 32)

        self.FPS = 30
        self.MAX_STEPS = 1800 # 60 seconds at 30 FPS
        self.MAX_GAME_TIME = 60.0
        self.CURSOR_SPEED = 10
        self.DNA_RADIUS = 12
        self.HOVER_RADIUS = 20
        self.MIN_SPAWN_DIST = self.DNA_RADIUS * 3
        self.INITIAL_DIFFICULTY = 3

        # State variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_dilation = None
        self.game_time = None
        self.cursor_pos = None
        self.space_was_held = None
        self.dna_segments = None
        self.selected_segment_id = None
        self.connections = None
        self.particles = None
        self.target_sequence = None
        self.current_synthesis_progress = None
        self.success_counter = 0
        self.current_difficulty = self.INITIAL_DIFFICULTY
        
        self.reset()
        # self.validate_implementation() # Removed for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_dilation = 1.0
        self.game_time = self.MAX_GAME_TIME
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.space_was_held = False
        self.selected_segment_id = None
        self.connections = []
        self.particles = []
        self.current_synthesis_progress = 0

        self.current_difficulty = self.INITIAL_DIFFICULTY + (self.success_counter // 3)
        self.current_difficulty = min(self.current_difficulty, 8) # Cap difficulty

        self._generate_target_sequence()
        self._generate_dna_segments()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._update_time_dilation(movement, shift_held)
        self._update_game_time()

        self._move_cursor(movement)

        space_pressed = space_held and not self.space_was_held
        if space_pressed:
            connection_reward = self._handle_connection()
            reward += connection_reward
        self.space_was_held = space_held

        self._update_particles()
        
        if self.current_synthesis_progress >= len(self.target_sequence):
            self.game_over = True
            reward += 100
            self.success_counter += 1
            # SFX: level_complete.wav

        if self.game_time <= 0:
            self.game_over = True
            reward -= 100
            # SFX: timeout_fail.wav

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False
        if self.steps >= self.MAX_STEPS and not self.game_over:
            reward -= 100 # Also penalize for running out of steps
            terminated = True

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
        self._render_background()
        self._render_connections()
        self._render_dna_segments()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "synthesis_progress": self.current_synthesis_progress,
            "target_length": len(self.target_sequence),
            "game_time": self.game_time,
        }

    # --- Game Logic Helpers ---

    def _generate_target_sequence(self):
        self.target_sequence = [random.choice(self.DNA_TYPES) for _ in range(self.current_difficulty)]

    def _generate_dna_segments(self):
        self.dna_segments = []
        # Ensure enough segments to complete the puzzle exist
        for dna_type in self.target_sequence:
            self._spawn_single_segment(dna_type)
        
        # Add complements to make pairs possible
        for dna_type in self.target_sequence:
             self._spawn_single_segment(self.DNA_COMPLEMENTS[dna_type])

        # Add random filler segments
        num_segments = 20 + self.current_difficulty * 2
        for _ in range(num_segments - len(self.dna_segments)):
            self._spawn_single_segment(random.choice(self.DNA_TYPES))

    def _spawn_single_segment(self, dna_type):
        for _ in range(100): # 100 attempts to find a free spot
            pos = pygame.Vector2(
                random.uniform(self.DNA_RADIUS * 2, self.WIDTH - self.DNA_RADIUS * 2),
                random.uniform(self.DNA_RADIUS * 2, self.HEIGHT - self.DNA_RADIUS * 2)
            )
            if not any(pos.distance_to(seg['pos']) < self.MIN_SPAWN_DIST for seg in self.dna_segments):
                break
        
        new_id = len(self.dna_segments)
        self.dna_segments.append({
            'id': new_id, 'pos': pos, 'type': dna_type, 
            'color': self.DNA_COLORS[dna_type], 'is_in_chain': False
        })

    def _update_time_dilation(self, movement, shift_held):
        if shift_held:
            self.time_dilation += 0.1
        elif movement == 0: # no-op
            self.time_dilation -= 0.1
        self.time_dilation = max(0.1, min(self.time_dilation, 5.0))

    def _update_game_time(self):
        if self.game_time > 0:
            self.game_time -= (1.0 / self.FPS) * self.time_dilation

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

    def _handle_connection(self):
        hovered_segment = self._get_hovered_segment()
        if not hovered_segment:
            if self.selected_segment_id is not None:
                self.selected_segment_id = None
                # SFX: deselect.wav
            return 0

        hovered_id = hovered_segment['id']
        if self.selected_segment_id is None:
            self.selected_segment_id = hovered_id
            # SFX: select_node.wav
            return 0
        else:
            if self.selected_segment_id == hovered_id:
                self.selected_segment_id = None
                return -0.01

            seg1 = self.dna_segments[self.selected_segment_id]
            seg2 = hovered_segment
            is_already_connected = tuple(sorted((seg1['id'], seg2['id']))) in self.connections
            
            self.selected_segment_id = None # Deselect after any attempt

            if is_already_connected:
                return -0.01

            if self.DNA_COMPLEMENTS[seg1['type']] == seg2['type']:
                # SFX: connection_success.wav
                self.connections.append(tuple(sorted((seg1['id'], seg2['id']))))
                self._create_particles(seg1['pos'].lerp(seg2['pos'], 0.5), seg1['color'], seg2['color'])
                progress_reward = self._check_synthesis_progress()
                return 0.1 + progress_reward
            else:
                # SFX: connection_fail.wav
                return -0.01

    def _check_synthesis_progress(self):
        for seg in self.dna_segments: seg['is_in_chain'] = False

        adj = {i: [] for i in range(len(self.dna_segments))}
        for u, v in self.connections:
            adj[u].append(v)
            adj[v].append(u)

        best_progress = 0
        best_path = []
        for i in range(len(self.dna_segments)):
            if self.dna_segments[i]['type'] == self.target_sequence[0]:
                path, progress = self._find_longest_path(i, adj)
                if progress > best_progress:
                    best_progress = progress
                    best_path = path
        
        old_progress = self.current_synthesis_progress
        self.current_synthesis_progress = best_progress
        
        for seg_id in best_path:
            self.dna_segments[seg_id]['is_in_chain'] = True

        return max(0, self.current_synthesis_progress - old_progress) * 1.0

    def _find_longest_path(self, start_node_id, adj):
        target = self.target_sequence
        stack = [(start_node_id, [start_node_id], 1)]
        max_progress = 0
        best_path = []

        if self.dna_segments[start_node_id]['type'] != target[0]: return [], 0
        max_progress, best_path = 1, [start_node_id]

        while stack:
            curr_id, path, progress = stack.pop()
            if progress > max_progress:
                max_progress, best_path = progress, path
            if progress >= len(target): continue

            for neighbor_id in adj[curr_id]:
                if neighbor_id not in path and self.dna_segments[neighbor_id]['type'] == target[progress]:
                    stack.append((neighbor_id, path + [neighbor_id], progress + 1))
        
        return best_path, max_progress

    def _get_hovered_segment(self):
        for seg in self.dna_segments:
            if self.cursor_pos.distance_to(seg['pos']) < self.HOVER_RADIUS:
                return seg
        return None

    # --- Particle System ---

    def _create_particles(self, pos, color1, color2):
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'lifetime': random.uniform(15, 30),
                'color': random.choice([color1, color2])
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    # --- Rendering Helpers ---

    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_connections(self):
        for id1, id2 in self.connections:
            seg1, seg2 = self.dna_segments[id1], self.dna_segments[id2]
            is_in_chain = seg1['is_in_chain'] and seg2['is_in_chain']
            color = self.COLOR_TEXT if is_in_chain else (100, 100, 120)
            width = 3 if is_in_chain else 1
            pygame.draw.aaline(self.screen, color, seg1['pos'], seg2['pos'], width)

    def _render_dna_segments(self):
        for seg in self.dna_segments:
            pos = (int(seg['pos'].x), int(seg['pos'].y))
            color = seg['color']
            radius = self.DNA_RADIUS

            if seg['is_in_chain']:
                glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, color + (60,), (radius * 2, radius * 2), radius * 1.8)
                self.screen.blit(glow_surf, (pos[0] - radius * 2, pos[1] - radius * 2), special_flags=pygame.BLEND_RGBA_ADD)

            if seg['id'] == self.selected_segment_id:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 3, self.COLOR_CURSOR)

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, tuple(c*0.8 for c in self.COLOR_TEXT))
            
            text = self.font_main.render(seg['type'], True, (0, 0, 0))
            self.screen.blit(text, text.get_rect(center=pos))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifetime'] / 30))))
            size = int(max(1, 5 * (p['lifetime'] / 30)))
            p_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(p_surf, p['color'] + (alpha,), (size, size), size)
            self.screen.blit(p_surf, (int(p['pos'].x - size), int(p['pos'].y - size)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_cursor(self):
        pos = (int(self.cursor_pos.x), int(self.cursor_pos.y))
        hovered = self._get_hovered_segment()
        if hovered and hovered['id'] != self.selected_segment_id:
            pygame.draw.circle(self.screen, self.COLOR_CURSOR, (int(hovered['pos'].x), int(hovered['pos'].y)), self.DNA_RADIUS + 7, 2)
        else:
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0] - 8, pos[1]), (pos[0] + 8, pos[1]), 2)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0], pos[1] - 8), (pos[0], pos[1] + 8), 2)

    def _render_ui(self):
        score_surf = self.font_main.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        time_str = f"Time: {max(0, self.game_time):.1f}s"
        time_surf = self.font_main.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, time_surf.get_rect(topright=(self.WIDTH - 10, 10)))

        dilation_str = f"Time Dilation: {self.time_dilation:.1f}x"
        dilation_surf = self.font_main.render(dilation_str, True, self.COLOR_TEXT)
        self.screen.blit(dilation_surf, dilation_surf.get_rect(bottomright=(self.WIDTH - 10, self.HEIGHT - 10)))

        title_surf = self.font_title.render("Target Sequence", True, self.COLOR_TEXT)
        self.screen.blit(title_surf, title_surf.get_rect(midtop=(self.WIDTH / 2, 5)))
        
        box_w, box_h, spacing = 30, 30, 5
        total_w = len(self.target_sequence) * (box_w + spacing) - spacing
        start_x = (self.WIDTH - total_w) / 2
        
        for i, dna_type in enumerate(self.target_sequence):
            rect = pygame.Rect(start_x + i * (box_w + spacing), 45, box_w, box_h)
            is_complete = i < self.current_synthesis_progress
            
            color = self.DNA_COLORS[dna_type] if is_complete else self.COLOR_GRID
            border_color = self.COLOR_TEXT if is_complete else (80, 80, 100)
            
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, border_color, rect, 2, border_radius=4)
            
            text_color = (0,0,0) if is_complete else self.COLOR_TEXT
            text_surf = self.font_main.render(dna_type, True, text_color)
            self.screen.blit(text_surf, text_surf.get_rect(center=rect.center))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # For manual play, we need a real display.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="human_playable")
    obs, info = env.reset()
    terminated = False
    
    # Override pygame screen for direct display
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("DNA Synthesizer")
    clock = pygame.time.Clock()

    while not terminated:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']:.2f}, Progress: {info['synthesis_progress']}/{info['target_length']}")

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            # Optional: wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()