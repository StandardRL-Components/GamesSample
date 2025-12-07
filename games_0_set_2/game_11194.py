import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:46:25.875958
# Source Brief: brief_01194.md
# Brief Index: 1194
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for 2D vectors to simplify physics calculations
class Vec2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2d(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2d(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        if scalar == 0: return Vec2d(0, 0)
        return Vec2d(self.x / scalar, self.y / scalar)

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        l = self.length()
        if l == 0: return Vec2d(0, 0)
        return self / l

    def to_int_tuple(self):
        return (int(self.x), int(self.y))

# Helper class for physics-based chromatin segments
class ChromatinSegment:
    def __init__(self, x, y, color, radius=8):
        self.pos = Vec2d(x, y)
        self.old_pos = Vec2d(x, y)
        self.color = color
        self.radius = radius

# Helper class for patrolling ribosomes
class Ribosome:
    def __init__(self, path_start, path_end, speed, radius=15):
        self.pos = Vec2d(path_start[0], path_start[1])
        self.path_start = Vec2d(path_start[0], path_start[1])
        self.path_end = Vec2d(path_end[0], path_end[1])
        self.speed = speed
        self.radius = radius
        self.direction = 1
        self.path_vector = self.path_end - self.path_start
        self.path_length = self.path_vector.length()
        self.path_unit_vector = self.path_vector.normalize()
        self.current_dist = 0

    def update(self, dt):
        # Speed is scaled by dt and a constant to make it feel right at 30 FPS
        self.current_dist += self.speed * self.direction * dt * 30
        if self.current_dist >= self.path_length:
            self.current_dist = self.path_length
            self.direction *= -1
        elif self.current_dist <= 0:
            self.current_dist = 0
            self.direction *= -1
        self.pos = self.path_start + self.path_unit_vector * self.current_dist

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Manipulate strands of chromatin with force fields to assemble a target genetic sequence. "
        "Avoid patrolling ribosomes that will disrupt your work."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Hold space to pull chromatin strands towards you, "
        "and hold shift to push them away."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 30
        self.NUCLEOTIDE_COLORS = [
            (255, 100, 100), # A - Red
            (100, 255, 100), # G - Green
            (100, 100, 255), # C - Blue
            (255, 255, 100), # T - Yellow
            (255, 100, 255), # U - Magenta
        ]
        self.COLOR_BG = (20, 40, 30)
        self.COLOR_RIBOSOME = (255, 50, 50)
        self.COLOR_TARGET_SLOT = (80, 80, 90)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_PULL = (255, 220, 0)
        self.COLOR_PUSH = (0, 200, 255)

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
        self.font_title = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Persistent State (across episodes) ---
        self.total_successes = 0
        self.base_ribosome_speed = 0.8

        # --- Game State (reset each episode) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = Vec2d(0, 0)
        self.strands = []
        self.ribosomes = []
        self.target_sequence = []
        self.target_slots = []
        self.last_segment_distances = {}
        self.last_action = [0, 0, 0]

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = Vec2d(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.last_action = [0, 0, 0]

        # --- Difficulty Scaling ---
        target_length = min(8, 3 + self.total_successes) # Cap length at 8
        num_ribosomes = min(4, 1 + self.total_successes // 3) # Cap ribosomes at 4
        ribosome_speed = self.base_ribosome_speed + (self.total_successes // 5) * 0.1

        # --- Generate Target Sequence ---
        self.target_sequence = [random.choice(self.NUCLEOTIDE_COLORS) for _ in range(target_length)]
        self.target_slots = []
        slot_y = 50
        slot_x_start = self.SCREEN_WIDTH - (target_length * 40) - 20
        for i in range(target_length):
            self.target_slots.append(Vec2d(slot_x_start + i * 40, slot_y))

        # --- Initialize Chromatin Strands ---
        self.strands = []
        all_colors = self.target_sequence[:]
        random.shuffle(all_colors)

        num_strands = max(1, target_length // 2)
        for i in range(num_strands):
            strand = []
            strand_len = len(all_colors) // num_strands + (1 if i < len(all_colors) % num_strands else 0)
            if strand_len == 0: continue
            
            start_pos = Vec2d(
                random.uniform(100, self.SCREEN_WIDTH - 100),
                random.uniform(100, self.SCREEN_HEIGHT - 100)
            )
            for j in range(strand_len):
                color = all_colors.pop()
                pos = start_pos + Vec2d(random.uniform(-10, 10), random.uniform(-10, 10))
                strand.append(ChromatinSegment(pos.x, pos.y, color))
            self.strands.append(strand)

        # --- Initialize Ribosomes ---
        self.ribosomes = []
        for _ in range(num_ribosomes):
            if random.random() > 0.5: # Horizontal
                y = random.uniform(80, self.SCREEN_HEIGHT - 30)
                start = (random.uniform(20, 50), y)
                end = (random.uniform(self.SCREEN_WIDTH - 50, self.SCREEN_WIDTH - 20), y)
            else: # Vertical
                x = random.uniform(20, self.SCREEN_WIDTH - 250)
                start = (x, random.uniform(80, 120))
                end = (x, random.uniform(self.SCREEN_HEIGHT - 120, self.SCREEN_HEIGHT - 30))
            if random.random() > 0.5: start, end = end, start
            self.ribosomes.append(Ribosome(start, end, ribosome_speed))
        
        self.last_segment_distances = self._get_segment_distances_to_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        self.last_action = action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0

        # 1. Update Cursor
        cursor_speed = 8
        if movement == 1: self.cursor_pos.y -= cursor_speed
        elif movement == 2: self.cursor_pos.y += cursor_speed
        elif movement == 3: self.cursor_pos.x -= cursor_speed
        elif movement == 4: self.cursor_pos.x += cursor_speed
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.SCREEN_WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.SCREEN_HEIGHT)

        # 2. Apply Forces
        force_radius = 80
        force_strength = 1.2
        if space_held or shift_held:
            # Sfx: hum_low.wav or hum_high.wav
            for strand in self.strands:
                for segment in strand:
                    dist_vec = self.cursor_pos - segment.pos
                    dist = dist_vec.length()
                    if 0 < dist < force_radius:
                        force_dir = dist_vec.normalize()
                        magnitude = force_strength * (1 - (dist / force_radius))
                        if space_held: force = force_dir * magnitude # Pull
                        else: force = force_dir * -magnitude # Push
                        segment.pos += force

        # 3. Update Physics
        self._update_physics(1.0 / self.FPS)
        for r in self.ribosomes:
            r.update(1.0 / self.FPS)

        # 4. Check for Termination & Calculate Rewards
        terminated = False
        
        # 4a. Collision with Ribosomes
        for r in self.ribosomes:
            for strand in self.strands:
                for segment in strand:
                    if (r.pos - segment.pos).length() < r.radius + segment.radius:
                        # Sfx: zap_error.wav
                        reward = -100
                        terminated = True
                        self.game_over = True
                        break
                if terminated: break
            if terminated: break

        # 4b. Sequence Completion
        if not terminated:
            correctly_placed, _ = self._check_sequence_completion()
            if correctly_placed == len(self.target_sequence):
                # Sfx: success_chime.wav
                reward = 100
                terminated = True
                self.game_over = True
                self.total_successes += 1
            else:
                reward += correctly_placed * 1.0

        # 4c. Max steps
        if self.steps >= 5000:
            terminated = True
            self.game_over = True

        # 4d. Continuous Rewards (if not terminal)
        if not terminated:
            current_distances = self._get_segment_distances_to_target()
            dist_reward = 0
            for color, dist in current_distances.items():
                prev_dist = self.last_segment_distances.get(color, dist)
                dist_reward += (prev_dist - dist) * 0.001
            self.last_segment_distances = current_distances
            reward += dist_reward
            
            proximity_penalty = 0
            for r in self.ribosomes:
                for strand in self.strands:
                    for seg in strand:
                        dist = (r.pos - seg.pos).length()
                        if dist < r.radius * 4:
                            proximity_penalty -= (1 / max(1, dist/r.radius)) * 0.01
            reward += proximity_penalty

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_physics(self, dt):
        friction = 0.99
        
        for strand in self.strands:
            for seg in strand:
                velocity = (seg.pos - seg.old_pos) * friction
                seg.old_pos = seg.pos
                seg.pos = seg.pos + velocity

        iterations = 5
        segment_length = 20
        for _ in range(iterations):
            for strand in self.strands:
                for seg in strand:
                    seg.pos.x = np.clip(seg.pos.x, seg.radius, self.SCREEN_WIDTH - seg.radius)
                    seg.pos.y = np.clip(seg.pos.y, seg.radius, self.SCREEN_HEIGHT - seg.radius)

            for strand in self.strands:
                for i in range(len(strand) - 1):
                    p1, p2 = strand[i], strand[i+1]
                    diff = p2.pos - p1.pos
                    dist = diff.length()
                    if dist == 0: continue
                    correction = (diff / dist) * (segment_length - dist) * 0.5
                    p1.pos -= correction
                    p2.pos += correction

    def _get_segment_distances_to_target(self):
        distances = {}
        all_segments = [seg for strand in self.strands for seg in strand]
        
        color_to_segments = {}
        for seg in all_segments:
            if seg.color not in color_to_segments:
                color_to_segments[seg.color] = []
            color_to_segments[seg.color].append(seg)

        for i, color in enumerate(self.target_sequence):
            target_pos = self.target_slots[i]
            if color not in color_to_segments:
                min_dist = self.SCREEN_WIDTH
            else:
                min_dist = min([(seg.pos - target_pos).length() for seg in color_to_segments[color]])
            
            key = (color, i) # Use tuple as key to distinguish same colors
            distances[key] = min_dist
        return distances

    def _check_sequence_completion(self):
        capture_radius = 15
        correctly_placed = 0
        total_placed = 0
        
        all_segments = [seg for strand in self.strands for seg in strand]
        placed_segments = set()

        for i, target_pos in enumerate(self.target_slots):
            target_color = self.target_sequence[i]
            for seg in all_segments:
                if id(seg) in placed_segments: continue
                if (seg.pos - target_pos).length() < capture_radius:
                    total_placed += 1
                    placed_segments.add(id(seg))
                    if seg.color == target_color:
                        correctly_placed += 1
                    break
        return correctly_placed, total_placed

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        for _ in range(30):
            pygame.gfxdraw.filled_circle(
                self.screen, random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT),
                random.randint(20, 60), (self.COLOR_BG[0]+10, self.COLOR_BG[1]+10, self.COLOR_BG[2]+5, 50)
            )

        correctly_placed, _ = self._check_sequence_completion()
        for i, pos in enumerate(self.target_slots):
            color = self.target_sequence[i]
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 15, self.COLOR_TARGET_SLOT)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 15, (*self.COLOR_TARGET_SLOT, 50))
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 5, color)

        for r in self.ribosomes:
            pygame.gfxdraw.filled_circle(self.screen, int(r.pos.x), int(r.pos.y), int(r.radius * 1.5), (*self.COLOR_RIBOSOME, 50))
            pygame.gfxdraw.filled_circle(self.screen, int(r.pos.x), int(r.pos.y), r.radius, self.COLOR_RIBOSOME)
            pygame.gfxdraw.aacircle(self.screen, int(r.pos.x), int(r.pos.y), r.radius, (0,0,0))
            pygame.gfxdraw.filled_circle(self.screen, int(r.pos.x+4), int(r.pos.y-4), 4, (255,150,150))

        for strand in self.strands:
            for i in range(len(strand) - 1):
                p1, p2 = strand[i].pos.to_int_tuple(), strand[i+1].pos.to_int_tuple()
                pygame.draw.aaline(self.screen, (200, 200, 220), p1, p2)
        for strand in self.strands:
            for seg in strand:
                pos = seg.pos.to_int_tuple()
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], seg.radius+2, (*seg.color, 80))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], seg.radius, seg.color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], seg.radius, (255,255,255))

        _, space_held, shift_held = self.last_action[0], self.last_action[1]==1, self.last_action[2]==1
        cursor_color = self.COLOR_CURSOR
        force_radius = 80
        if space_held: cursor_color = self.COLOR_PULL
        elif shift_held: cursor_color = self.COLOR_PUSH
        pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), force_radius, (*cursor_color, 80))
        pygame.gfxdraw.filled_circle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 5, cursor_color)
        pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 5, (0,0,0))
        
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, (220, 220, 255))
        self.screen.blit(score_text, (10, 10))
        level_text = self.font_ui.render(f"Level: {self.total_successes + 1}", True, (220, 220, 255))
        self.screen.blit(level_text, (10, 30))
        target_label = self.font_title.render("Target Sequence", True, (220, 220, 255))
        self.screen.blit(target_label, (self.SCREEN_WIDTH - 250, 10))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.total_successes + 1,
            "target_length": len(self.target_sequence),
        }

    def render(self):
        if not hasattr(self, 'window') or self.window is None:
             pygame.display.init()
             self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
             pygame.display.set_caption("Chromatin Synthesis")
        
        obs_frame = self._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(obs_frame, (1, 0, 2)))
        self.window.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def close(self):
        if hasattr(self, 'window') and self.window is not None:
            pygame.display.quit()
            self.window = None
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        self.last_action = test_action
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    env.render() # To create the window
    obs, info = env.reset()
    done = False
    
    key_map = {
        pygame.K_UP: 1, pygame.K_w: 1,
        pygame.K_DOWN: 2, pygame.K_s: 2,
        pygame.K_LEFT: 3, pygame.K_a: 3,
        pygame.K_RIGHT: 4, pygame.K_d: 4,
    }

    print("\n--- Human Controls ---")
    print("Arrow Keys/WASD: Move cursor")
    print("Spacebar: Pull chromatin")
    print("Shift: Push chromatin")
    print("R: Reset environment")
    print("Q: Quit")

    while not done:
        movement_action, space_action, shift_action = 0, 0, 0
        
        # This loop is for event handling (like closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: 
                    obs, info = env.reset()
                    print(f"Environment reset. Level: {info['level']}")
                if event.key == pygame.K_q: done = True
        if done: break

        # This part checks for currently held keys for continuous actions
        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']:.1f}, Level: {info['level']}")
            # Wait for R or Q
            end_loop = True
            while end_loop:
                event = pygame.event.wait()
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    done = True
                    end_loop = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    print(f"Environment reset. Level: {info['level']}")
                    end_loop = False
            if done: break
            
    env.close()