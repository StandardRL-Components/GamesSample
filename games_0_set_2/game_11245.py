import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:43:41.896633
# Source Brief: brief_01245.md
# Brief Index: 1245
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Fractal Firewall: A cyberpunk-themed puzzle/action game.

    The player must defend a firewall grid against incoming cyberattacks. Each
    attack is a projectile carrying a specific fractal pattern. The player selects
    a cell on the firewall grid, which also contains a fractal pattern. When an
    attacker hits the firewall, its pattern is compared to the currently selected
    cell's pattern.

    - A match neutralizes the attack and rewards the player.
    - A mismatch damages the firewall.

    The player can manipulate time, slowing it down to plan or speeding it up
    for greater risk and reward. The game progresses in waves of increasing
    difficulty.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your firewall by matching incoming fractal attacks. Select the correct fractal on "
        "your grid to neutralize threats and survive waves of increasing difficulty."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selector. Hold space to slow down time or shift to speed it up."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    GRID_DIMS = (5, 5)
    GRID_CELL_SIZE = 55
    GRID_SPACING = 10
    FIREWALL_Y_POS = 320
    MAX_STEPS = 5000

    # --- COLORS (Cyberpunk Palette) ---
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (30, 60, 90)
    COLOR_SELECTOR = (0, 255, 255)
    COLOR_DEFENSE_FRACTAL = (50, 200, 255)
    COLOR_ATTACK_FRACTAL = (255, 100, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SUCCESS = (255, 255, 150)
    COLOR_DAMAGE = (255, 50, 50)
    COLOR_TIME_SLOW = (100, 150, 255, 50)
    COLOR_TIME_FAST = (255, 150, 100, 50)


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_wave = pygame.font.Font(None, 48)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 0
        self.firewall_integrity = 100.0
        self.time_factor = 1.0
        self.selector_pos = [0, 0]
        self.move_cooldown = 0

        self.grid_fractals = []
        self.fractal_definitions = {}
        self.unique_fractal_count = 0

        self.attackers = []
        self.attacks_to_spawn = []
        self.next_attack_timer = 0
        self.wave_transition_timer = 0

        self.particles = []
        self.background_stars = []

        self.grid_start_x = (self.WIDTH - (self.GRID_DIMS[0] * self.GRID_CELL_SIZE + (self.GRID_DIMS[0] - 1) * self.GRID_SPACING)) // 2
        self.grid_start_y = self.FIREWALL_Y_POS - self.GRID_CELL_SIZE - 20

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.firewall_integrity = 100.0
        self.time_factor = 1.0
        self.selector_pos = [self.GRID_DIMS[0] // 2, self.GRID_DIMS[1] // 2]
        self.move_cooldown = 0
        self.wave = 0
        self.particles = []
        self.attackers = []
        self.wave_transition_timer = 120 # 4s transition

        self._init_background()
        self._start_new_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        self._handle_input(action)

        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1
        else:
            self._update_game_logic()
            reward += self._process_events()

        self._update_particles()

        # Continuous reward for survival
        if self.firewall_integrity > 0:
            reward += 0.01

        terminated = self.firewall_integrity <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            if self.firewall_integrity <= 0:
                reward -= 100  # Large penalty for failure
                # // SFX: System failure alarm
            self.game_over = True

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Time manipulation: Slowdown has priority
        if space_held:
            self.time_factor = 0.5
        elif shift_held:
            self.time_factor = 2.0
        else:
            self.time_factor = 1.0

        # Selector movement with cooldown
        if self.move_cooldown == 0 and movement != 0:
            if movement == 1: self.selector_pos[1] -= 1 # Up
            elif movement == 2: self.selector_pos[1] += 1 # Down
            elif movement == 3: self.selector_pos[0] -= 1 # Left
            elif movement == 4: self.selector_pos[0] += 1 # Right
            self.selector_pos[0] = np.clip(self.selector_pos[0], 0, self.GRID_DIMS[0] - 1)
            self.selector_pos[1] = np.clip(self.selector_pos[1], 0, self.GRID_DIMS[1] - 1)
            self.move_cooldown = 5 # Cooldown of 5 steps
        
        if self.move_cooldown > 0:
            self.move_cooldown -= 1

    def _update_game_logic(self):
        # Spawn new attackers
        self.next_attack_timer -= self.time_factor
        if self.next_attack_timer <= 0 and self.attacks_to_spawn:
            self._spawn_attacker()
            spawn_interval = self.np_random.uniform(90, 150) / max(1, self.wave * 0.5)
            self.next_attack_timer = spawn_interval

        # Move attackers
        for attacker in self.attackers:
            attacker["pos"][1] += attacker["speed"] * self.time_factor

    def _process_events(self):
        reward = 0
        
        # Check for attacker hits
        for attacker in self.attackers[:]:
            if attacker["pos"][1] >= self.FIREWALL_Y_POS:
                selected_fractal_id = self.grid_fractals[self.selector_pos[1]][self.selector_pos[0]]
                
                if attacker["fractal_id"] == selected_fractal_id:
                    # Successful match
                    reward += 5
                    self.score += 100 + int(self.wave * 10)
                    self._create_particles(attacker["pos"], 30, self.COLOR_SUCCESS, 2.5)
                    # // SFX: Success chime, chain reaction sizzle
                else:
                    # Failed match
                    reward -= 2
                    self.firewall_integrity -= 10
                    self.firewall_integrity = max(0, self.firewall_integrity)
                    self._create_particles(attacker["pos"], 50, self.COLOR_DAMAGE, 3.5)
                    # // SFX: Error buzz, shield impact
                
                self.attackers.remove(attacker)

        # Check for wave clear
        if not self.attackers and not self.attacks_to_spawn:
            reward += 50
            self.score += self.wave * 250
            self.wave_transition_timer = 120
            self._start_new_wave()
            # // SFX: Wave complete fanfare

        return reward

    def _start_new_wave(self):
        self.wave += 1
        
        # Difficulty scaling
        attack_speed = min(2.5, 0.8 + self.wave * 0.1)
        fractal_complexity = min(5, 2 + self.wave // 3)
        self.unique_fractal_count = min(8, 3 + self.wave // 2)
        num_attacks = 5 + self.wave * 2

        # Generate fractal definitions for this wave
        self.fractal_definitions.clear()
        for i in range(self.unique_fractal_count):
            seed = self.np_random.integers(1000)
            self.fractal_definitions[i] = self._generate_fractal_pattern(seed, fractal_complexity)

        # Create attack sequence
        self.attacks_to_spawn = self.np_random.integers(0, self.unique_fractal_count, size=num_attacks).tolist()

        # Populate grid ensuring all needed fractals are available
        self.grid_fractals = [[0] * self.GRID_DIMS[0] for _ in range(self.GRID_DIMS[1])]
        needed_fractals = list(set(self.attacks_to_spawn))
        
        # Place required fractals
        available_cells = [(r, c) for r in range(self.GRID_DIMS[1]) for c in range(self.GRID_DIMS[0])]
        self.np_random.shuffle(available_cells)
        
        for i, fractal_id in enumerate(needed_fractals):
            if i < len(available_cells):
                r, c = available_cells.pop()
                self.grid_fractals[r][c] = fractal_id
        
        # Fill remaining cells randomly
        for r, c in available_cells:
            self.grid_fractals[r][c] = self.np_random.integers(0, self.unique_fractal_count)

        self.next_attack_timer = 120 # Initial delay for the first attacker

    def _spawn_attacker(self):
        if not self.attacks_to_spawn:
            return

        fractal_id = self.attacks_to_spawn.pop(0)
        speed = min(2.5, 0.8 + self.wave * 0.1) * self.np_random.uniform(0.9, 1.1)
        
        attacker = {
            "pos": [self.np_random.uniform(50, self.WIDTH - 50), -20.0],
            "fractal_id": fractal_id,
            "speed": speed,
            "angle": self.np_random.uniform(0, 2 * math.pi)
        }
        self.attackers.append(attacker)
        # // SFX: Attacker spawn warp

    def _generate_fractal_pattern(self, seed, complexity):
        rng = random.Random(seed)
        lines = []
        
        def _recurse(x, y, angle, length, depth):
            if depth == 0:
                return
            
            nx = x + length * math.cos(angle)
            ny = y + length * math.sin(angle)
            lines.append(((x, y), (nx, ny)))
            
            branch_angle = rng.uniform(0.4, 1.2)
            length_scale = rng.uniform(0.6, 0.8)
            
            _recurse(nx, ny, angle - branch_angle, length * length_scale, depth - 1)
            _recurse(nx, ny, angle + branch_angle, length * length_scale, depth - 1)

        _recurse(0, 0, -math.pi / 2, 10, complexity)
        return lines

    def _init_background(self):
        self.background_stars = []
        for _ in range(150):
            self.background_stars.append({
                "pos": [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)],
                "depth": self.np_random.uniform(0.1, 0.6),
                "brightness": self.np_random.uniform(50, 150)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 0.02
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": 1.0,
                "color": color,
                "radius": self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "firewall_integrity": self.firewall_integrity,
        }

    def _render_background(self):
        for star in self.background_stars:
            star["pos"][1] = (star["pos"][1] + star["depth"] * self.time_factor) % self.HEIGHT
            b = star["brightness"]
            pygame.draw.circle(self.screen, (b, b, b), star["pos"], star["depth"] * 1.5)

    def _render_game(self):
        # Render firewall grid
        for r in range(self.GRID_DIMS[1]):
            for c in range(self.GRID_DIMS[0]):
                x = self.grid_start_x + c * (self.GRID_CELL_SIZE + self.GRID_SPACING)
                y = self.grid_start_y + r * (self.GRID_CELL_SIZE + self.GRID_SPACING)
                rect = pygame.Rect(x, y, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1, border_radius=4)
                
                fractal_id = self.grid_fractals[r][c]
                self._render_fractal(self.screen, self.fractal_definitions[fractal_id], (x + self.GRID_CELL_SIZE/2, y + self.GRID_CELL_SIZE/2 + 15), 1.5, self.COLOR_DEFENSE_FRACTAL)

        # Render selector
        sel_x = self.grid_start_x + self.selector_pos[0] * (self.GRID_CELL_SIZE + self.GRID_SPACING)
        sel_y = self.grid_start_y + self.selector_pos[1] * (self.GRID_CELL_SIZE + self.GRID_SPACING)
        sel_rect = pygame.Rect(sel_x - 4, sel_y - 4, self.GRID_CELL_SIZE + 8, self.GRID_CELL_SIZE + 8)
        
        # Glow effect for selector
        glow_alpha = 100 + 50 * math.sin(self.steps * 0.2)
        for i in range(4):
            glow_rect = sel_rect.inflate(i*2, i*2)
            pygame.draw.rect(self.screen, (*self.COLOR_SELECTOR, glow_alpha / (i+1)), glow_rect, 1, border_radius=6)
        
        # Render attackers
        for attacker in self.attackers:
            pos = attacker["pos"]
            attacker["angle"] += 0.02 * self.time_factor
            self._render_fractal(self.screen, self.fractal_definitions[attacker["fractal_id"]], pos, 2.0, self.COLOR_ATTACK_FRACTAL, attacker["angle"])
            
            # Glow effect for attacker core
            for i in range(3):
                alpha = 150 / (i + 1)
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 8 + i*2, (*self.COLOR_ATTACK_FRACTAL, alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 8, self.COLOR_ATTACK_FRACTAL)

        # Render particles
        for p in self.particles:
            alpha = max(0, p["life"] * 255)
            pygame.gfxdraw.aacircle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), (*p["color"], alpha))

    def _render_fractal(self, surface, lines, pos, scale, color, angle=0):
        for line in lines:
            p1_x = line[0][0] * scale
            p1_y = line[0][1] * scale
            p2_x = line[1][0] * scale
            p2_y = line[1][1] * scale

            # Apply rotation
            r_p1_x = p1_x * math.cos(angle) - p1_y * math.sin(angle)
            r_p1_y = p1_x * math.sin(angle) + p1_y * math.cos(angle)
            r_p2_x = p2_x * math.cos(angle) - p2_y * math.sin(angle)
            r_p2_y = p2_x * math.sin(angle) + p2_y * math.cos(angle)

            start_pos = (pos[0] + r_p1_x, pos[1] + r_p1_y)
            end_pos = (pos[0] + r_p2_x, pos[1] + r_p2_y)
            pygame.draw.aaline(surface, color, start_pos, end_pos)

    def _render_ui(self):
        # Score and Wave
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        wave_text = self.font_ui.render(f"WAVE: {self.wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        # Firewall integrity bar and text
        integrity_perc = self.firewall_integrity / 100.0
        bar_width = 200
        bar_height = 15
        bar_x = (self.WIDTH - bar_width) // 2
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), 1)
        
        integrity_color = (
            (1 - integrity_perc) * 255,
            integrity_perc * 200,
            integrity_perc * 255
        )
        pygame.draw.rect(self.screen, integrity_color, (bar_x, bar_y, bar_width * integrity_perc, bar_height))
        
        # Integrity as glowing border
        if integrity_perc > 0:
            border_points = [
                (0, 0), (self.WIDTH, 0), (self.WIDTH, self.HEIGHT), (0, self.HEIGHT)
            ]
            total_len = (self.WIDTH + self.HEIGHT) * 2
            current_len = total_len * integrity_perc
            
            points_to_draw = []
            len_so_far = 0
            
            p1 = border_points[0]
            points_to_draw.append(p1)
            for i in range(4):
                p2 = border_points[(i + 1) % 4]
                segment_len = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                if len_so_far + segment_len < current_len:
                    points_to_draw.append(p2)
                    len_so_far += segment_len
                else:
                    rem_len = current_len - len_so_far
                    interp = rem_len / segment_len
                    final_p = (p1[0] + (p2[0] - p1[0]) * interp, p1[1] + (p2[1] - p1[1]) * interp)
                    points_to_draw.append(final_p)
                    break
                p1 = p2
            
            if len(points_to_draw) > 1:
                pygame.draw.aalines(self.screen, integrity_color, False, points_to_draw, 2)

        # Time manipulation overlay
        if self.time_factor != 1.0:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            color = self.COLOR_TIME_SLOW if self.time_factor < 1.0 else self.COLOR_TIME_FAST
            overlay.fill(color)
            self.screen.blit(overlay, (0, 0))

        # Wave transition text
        if self.wave_transition_timer > 0:
            alpha = min(255, self.wave_transition_timer * 4)
            wave_surf = self.font_wave.render(f"WAVE {self.wave}", True, self.COLOR_TEXT)
            wave_surf.set_alpha(alpha)
            self.screen.blit(wave_surf, (self.WIDTH/2 - wave_surf.get_width()/2, self.HEIGHT/2 - wave_surf.get_height()/2))
            
    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == '__main__':
    # This block will not run in the test environment, but is useful for local development.
    # It requires a display to be available.
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Human Play Controls ---
    # Arrows: Move selector
    # Space: Slow time
    # Left Shift: Speed up time
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Fractal Firewall")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS for human play
        
    env.close()