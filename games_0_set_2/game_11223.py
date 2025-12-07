import gymnasium as gym
import os
import pygame
import numpy as np
import math
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An action-puzzle Gymnasium environment where the agent solves 3D geometry
    problems to charge an attack, defeating waves of enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Solve geometry puzzles about the rotating cube to charge a powerful area-of-effect attack. "
        "Defeat waves of incoming enemies before you are overwhelmed."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to change a digit and ↑↓ to select which digit to change. "
        "Press space to submit your answer and shift to clear your input."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 48)
        
        # --- Colors ---
        self.COLOR_BG = (15, 18, 33)
        self.COLOR_GRID = (30, 35, 60)
        self.COLOR_CUBE_EDGES = (200, 200, 255)
        self.COLOR_CUBE_GLOW = (100, 100, 255, 30)
        self.COLOR_ENEMY = (255, 200, 0)
        self.COLOR_ENEMY_GLOW = (255, 200, 0, 50)
        self.COLOR_ATTACK_WAVE = (0, 191, 255)
        self.COLOR_ATTACK_WAVE_GLOW = (0, 191, 255, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_METER_BG = (40, 40, 70)
        self.COLOR_METER_FILL = (0, 191, 255)
        self.COLOR_CORRECT = (0, 255, 127)
        self.COLOR_INCORRECT = (255, 69, 0)
        self.COLOR_INPUT_ACTIVE = (255, 255, 255)
        self.COLOR_INPUT_INACTIVE = (128, 128, 128)

        # --- Game Constants ---
        self.MAX_STEPS = 1000
        self.WIN_CONDITION_ENEMIES = 50
        self.CENTER = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2 - 20)
        self.ATTACK_CHARGE_PER_SOLVE = 0.34 # 3 solves to full charge
        self.INITIAL_SPAWN_RATE = 0.1
        self.SPAWN_RATE_INCREASE = 0.01

        # --- State Variables ---
        # These are all initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.enemies_defeated = 0
        self.enemies = []
        self.particles = []
        self.attack_meter = 0.0
        self.spawn_rate = 0.0
        self.attack_wave = None
        self.player_input = []
        self.selected_digit_index = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.feedback_flash = {}
        self.cube_angle_x = 0
        self.cube_angle_y = 0
        self.cube_angle_z = 0
        self.current_puzzle = {}
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.enemies_defeated = 0
        self.enemies = []
        self.particles = []
        self.attack_meter = 0.0
        self.spawn_rate = self.INITIAL_SPAWN_RATE
        self.attack_wave = None
        self.last_space_held = False
        self.last_shift_held = False
        self.feedback_flash = {"color": self.COLOR_BG, "timer": 0}
        
        self.cube_angle_x = self.np_random.uniform(0, 2 * math.pi)
        self.cube_angle_y = self.np_random.uniform(0, 2 * math.pi)
        self.cube_angle_z = self.np_random.uniform(0, 2 * math.pi)

        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        # Map actions to puzzle input
        if movement == 1: # Up
            self.selected_digit_index = max(0, self.selected_digit_index - 1)
        elif movement == 2: # Down
            self.selected_digit_index = min(len(self.player_input) - 1, self.selected_digit_index + 1)
        elif movement == 3: # Left
            self.player_input[self.selected_digit_index] = (self.player_input[self.selected_digit_index] - 1) % 10
        elif movement == 4: # Right
            self.player_input[self.selected_digit_index] = (self.player_input[self.selected_digit_index] + 1) % 10
        
        if shift_pressed:
            self.player_input = [0] * len(self.player_input)
            self.selected_digit_index = 0

        if space_pressed:
            submitted_answer = int("".join(map(str, self.player_input)))
            if submitted_answer == self.current_puzzle["answer"]:
                reward += 0.1
                self.score += 10
                self.attack_meter = min(1.0, self.attack_meter + self.ATTACK_CHARGE_PER_SOLVE)
                self.feedback_flash = {"color": self.COLOR_CORRECT, "timer": 15}
                self._generate_puzzle()
            else:
                self.feedback_flash = {"color": self.COLOR_INCORRECT, "timer": 15}

        # --- Update Game State ---
        self._update_cube()
        self._update_enemies()
        reward += self._update_attack_wave()
        self._update_particles()
        
        if self.steps % 50 == 0:
            self.spawn_rate += self.SPAWN_RATE_INCREASE
            
        if self.feedback_flash["timer"] > 0:
            self.feedback_flash["timer"] -= 1

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.enemies_defeated >= self.WIN_CONDITION_ENEMIES:
            reward += 100
            self.score += 1000
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_puzzle(self):
        puzzle_type = self.np_random.choice(["Volume", "Surface Area", "Edge Length"])
        side = self.np_random.integers(2, 10)
        
        if puzzle_type == "Volume":
            self.current_puzzle = {
                "question": f"Side = {side}, Volume = ?",
                "answer": side ** 3
            }
        elif puzzle_type == "Surface Area":
            self.current_puzzle = {
                "question": f"Side = {side}, Surface Area = ?",
                "answer": 6 * (side ** 2)
            }
        else: # Edge Length
            self.current_puzzle = {
                "question": f"Side = {side}, Total Edge Length = ?",
                "answer": 12 * side
            }
        
        num_digits = len(str(self.current_puzzle["answer"]))
        self.player_input = [0] * num_digits
        self.selected_digit_index = 0

    def _update_cube(self):
        self.cube_angle_x += 0.01
        self.cube_angle_y += 0.015
        self.cube_angle_z += 0.005

    def _update_enemies(self):
        # Spawn new enemies
        if self.np_random.random() < self.spawn_rate:
            angle = self.np_random.uniform(0, 2 * math.pi)
            spawn_dist = max(self.WIDTH, self.HEIGHT) / 2 + 20
            pos = self.CENTER + pygame.Vector2(math.cos(angle), math.sin(angle)) * spawn_dist
            self.enemies.append({"pos": pos, "vel": (self.CENTER - pos).normalize() * 1.5})
        
        # Move existing enemies
        for enemy in self.enemies:
            enemy["pos"] += enemy["vel"]

    def _update_attack_wave(self):
        reward = 0
        if self.attack_meter >= 1.0 and self.attack_wave is None:
            self.attack_meter = 0.0
            self.attack_wave = {"radius": 0, "max_radius": 300, "speed": 10, "lifetime": 30}
            reward += 1.0
            
        if self.attack_wave:
            self.attack_wave["radius"] += self.attack_wave["speed"]
            self.attack_wave["lifetime"] -= 1
            
            enemies_hit_this_frame = []
            for enemy in self.enemies:
                if self.CENTER.distance_to(enemy["pos"]) <= self.attack_wave["radius"]:
                    enemies_hit_this_frame.append(enemy)
            
            if len(enemies_hit_this_frame) >= 10:
                reward += 5.0

            for enemy in enemies_hit_this_frame:
                if enemy in self.enemies:
                    self.enemies.remove(enemy)
                    self.enemies_defeated += 1
                    self.score += 50
                    self._create_explosion(enemy["pos"])

            if self.attack_wave["lifetime"] <= 0 or self.attack_wave["radius"] >= self.attack_wave["max_radius"]:
                self.attack_wave = None
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": self.COLOR_ENEMY,
                "size": self.np_random.integers(2, 5)
            })

    def _get_observation(self):
        flash_color = self.feedback_flash["color"] if self.feedback_flash["timer"] > 0 else self.COLOR_BG
        self.screen.fill(flash_color)
        
        self._render_background()
        self._render_particles()
        self._render_enemies()
        self._render_attack_wave()
        self._render_cube()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "enemies_defeated": self.enemies_defeated,
            "attack_meter": self.attack_meter,
            "puzzle_answer": self.current_puzzle.get("answer", 0)
        }

    def _render_background(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

    def _render_cube(self):
        scale = 80
        points = [
            np.array([-1, -1, 1]), np.array([1, -1, 1]), np.array([1, 1, 1]), np.array([-1, 1, 1]),
            np.array([-1, -1, -1]), np.array([1, -1, -1]), np.array([1, 1, -1]), np.array([-1, 1, -1])
        ]
        
        # Rotation matrices
        rot_x = np.array([[1, 0, 0], [0, math.cos(self.cube_angle_x), -math.sin(self.cube_angle_x)], [0, math.sin(self.cube_angle_x), math.cos(self.cube_angle_x)]])
        rot_y = np.array([[math.cos(self.cube_angle_y), 0, math.sin(self.cube_angle_y)], [0, 1, 0], [-math.sin(self.cube_angle_y), 0, math.cos(self.cube_angle_y)]])
        rot_z = np.array([[math.cos(self.cube_angle_z), -math.sin(self.cube_angle_z), 0], [math.sin(self.cube_angle_z), math.cos(self.cube_angle_z), 0], [0, 0, 1]])
        
        projected_points = []
        for point in points:
            rotated = np.dot(rot_z, np.dot(rot_y, np.dot(rot_x, point)))
            projected = [int(rotated[0] * scale + self.CENTER.x), int(rotated[1] * scale + self.CENTER.y)]
            projected_points.append(projected)

        # Draw glow
        for p in projected_points:
            pygame.gfxdraw.filled_circle(self.screen, p[0], p[1], 12, self.COLOR_CUBE_GLOW)
        
        # Draw edges
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for p1_idx, p2_idx in edges:
            p1 = projected_points[p1_idx]
            p2 = projected_points[p2_idx]
            pygame.draw.aaline(self.screen, self.COLOR_CUBE_EDGES, p1, p2)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
    
    def _render_attack_wave(self):
        if self.attack_wave:
            radius = int(self.attack_wave["radius"])
            alpha = int(255 * (self.attack_wave["lifetime"] / 30))
            color = self.COLOR_ATTACK_WAVE[:3] + (alpha,)
            glow_color = self.COLOR_ATTACK_WAVE_GLOW[:3] + (int(self.COLOR_ATTACK_WAVE_GLOW[3] * (alpha/255)),)
            
            if radius > 0:
                # Outer glow
                pygame.gfxdraw.filled_circle(self.screen, int(self.CENTER.x), int(self.CENTER.y), radius, glow_color)
                # Inner solid line
                pygame.gfxdraw.aacircle(self.screen, int(self.CENTER.x), int(self.CENTER.y), radius, color)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * p["life"] / 30.0)))
            color = p["color"][:3] + (alpha,)
            pygame.draw.circle(self.screen, color, (int(p["pos"].x), int(p["pos"].y)), int(p["size"]))

    def _render_ui(self):
        # --- Top UI ---
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        enemies_text = self.font_medium.render(f"Defeated: {self.enemies_defeated} / {self.WIN_CONDITION_ENEMIES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(enemies_text, (10, 10))

        # --- Puzzle UI ---
        puzzle_text = self.font_large.render(self.current_puzzle["question"], True, self.COLOR_UI_TEXT)
        text_rect = puzzle_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 120))
        self.screen.blit(puzzle_text, text_rect)

        # --- Player Input Display ---
        input_str = "".join(map(str, self.player_input))
        digit_width = 30
        total_width = len(input_str) * digit_width
        start_x = self.WIDTH / 2 - total_width / 2

        for i, digit in enumerate(input_str):
            color = self.COLOR_INPUT_ACTIVE if i == self.selected_digit_index else self.COLOR_INPUT_INACTIVE
            digit_surf = self.font_large.render(digit, True, color)
            digit_rect = digit_surf.get_rect(center=(start_x + i * digit_width + digit_width/2, self.HEIGHT - 80))
            self.screen.blit(digit_surf, digit_rect)
            if i == self.selected_digit_index:
                 pygame.draw.rect(self.screen, self.COLOR_INPUT_ACTIVE, (digit_rect.left - 2, digit_rect.bottom, digit_rect.width + 4, 2))

        # --- Attack Meter ---
        meter_width, meter_height = 400, 20
        meter_x, meter_y = self.WIDTH/2 - meter_width/2, self.HEIGHT - 40
        pygame.draw.rect(self.screen, self.COLOR_METER_BG, (meter_x, meter_y, meter_width, meter_height), border_radius=5)
        fill_width = meter_width * self.attack_meter
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_METER_FILL, (meter_x, meter_y, fill_width, meter_height), border_radius=5)
        
        meter_text = self.font_small.render(f"ATTACK: {int(self.attack_meter * 100)}%", True, self.COLOR_UI_TEXT)
        meter_text_rect = meter_text.get_rect(center=(self.WIDTH/2, meter_y + meter_height/2))
        self.screen.blit(meter_text, meter_text_rect)

    def close(self):
        pygame.quit()


# Example usage:
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Setup a window for human play
    pygame.display.set_caption("Geometry Attack")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = [0, 0, 0] # [movement, space, shift]
    
    while not done:
        # Action is based on keys held down
        keys = pygame.key.get_pressed()
        action[0] = 0 # No-op by default
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # Movement is event-based to avoid rapid changes
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

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()