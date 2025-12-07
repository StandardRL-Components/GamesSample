import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import math
import os
import pygame


# Pygame must run headless for the environment.
# The main execution block will handle display initialization for human play.
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a shrinking, color-changing circle.
    The goal is to navigate the circle through a series of color-matched gates
    before the circle shrinks to nothing.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        - Up/Down: Change distance from the center (orbit radius).
        - Left/Right: Change rotational speed.
    - actions[1]: Space button (0=released, 1=held) - Cycles color forward on press.
    - actions[2]: Shift button (0=released, 1=held) - Cycles color backward on press.

    Observation Space: Box(400, 640, 3) - An RGB image of the game screen.

    Reward Structure:
    - +10 for passing through a correct gate.
    - +100 for winning (passing all gates).
    - -5 for colliding with an incorrect gate.
    - -100 for losing (shrinking to nothing or wrong collision).
    - +/-0.1 for moving towards/away from the target gate.
    """
    game_description = "Control a shrinking circle and navigate through a series of color-matched gates before it disappears."
    user_guide = "Use ↑↓ arrow keys to change orbit radius and ←→ to change speed. Use Space/Shift to cycle your color to match the target gate."
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    MAX_STEPS = 1500 # 50 seconds at 30 FPS

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_WHITE = (240, 240, 240)
    COLOR_PALETTE = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255),   # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
    ]
    COLOR_NAMES = ["RED", "GREEN", "BLUE", "YELLOW", "MAGENTA"]

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_orbit_radius = 0.0
        self.player_angle = 0.0
        self.player_angular_velocity = 0.0
        self.player_size = 0.0
        self.player_color_index = 0

        self.gates = []
        self.target_gate_order = []
        self.current_target_idx = 0

        self.shrink_timer = 0
        self.shrink_interval = 10 * self.FPS # 10 seconds

        self.particles = []
        self.stars = []

        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Player state
        self.player_orbit_radius = 100.0  # Start at a safe distance
        self.player_angle = 0.0
        self.player_angular_velocity = 0.01
        self.player_size = 30.0
        self.player_color_index = self.np_random.integers(0, len(self.COLOR_PALETTE))

        # Gate setup
        self.gates.clear()
        gate_orbit_radius = self.SCREEN_HEIGHT / 2 - 30
        for i in range(len(self.COLOR_PALETTE)):
            angle = (2 * math.pi / len(self.COLOR_PALETTE)) * i
            x = self.CENTER[0] + gate_orbit_radius * math.cos(angle)
            y = self.CENTER[1] + gate_orbit_radius * math.sin(angle)
            self.gates.append({
                "pos": (x, y),
                "angle": angle,
                "color_index": i,
                "width": 80
            })
        
        self.target_gate_order = list(range(len(self.COLOR_PALETTE)))
        self.np_random.shuffle(self.target_gate_order)
        self.current_target_idx = 0

        # Timers and effects
        self.shrink_timer = self.shrink_interval
        self.particles.clear()
        self._create_stars()

        # Input state
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # On subsequent steps after termination, return the final state
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False

        # --- 1. Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement actions
        if movement == 1: # Up
            self.player_orbit_radius += 2.0
        elif movement == 2: # Down
            self.player_orbit_radius -= 2.0
        elif movement == 3: # Left
            self.player_angular_velocity -= 0.002
        elif movement == 4: # Right
            self.player_angular_velocity += 0.002
        
        # Clamp player values
        self.player_orbit_radius = max(10, min(self.player_orbit_radius, self.SCREEN_HEIGHT / 2 - 10))
        self.player_angular_velocity = max(-0.05, min(self.player_angular_velocity, 0.05))

        # Color change actions (on press)
        if space_held and not self.prev_space_held:
            self.player_color_index = (self.player_color_index + 1) % len(self.COLOR_PALETTE)
            self._create_particles(self._get_player_pos(), self.COLOR_PALETTE[self.player_color_index], 30, 3)
        if shift_held and not self.prev_shift_held:
            self.player_color_index = (self.player_color_index - 1 + len(self.COLOR_PALETTE)) % len(self.COLOR_PALETTE)
            self._create_particles(self._get_player_pos(), self.COLOR_PALETTE[self.player_color_index], 30, 3)

        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # --- 2. Update Game Logic & Physics ---
        old_dist = self._dist_to_target()
        
        self.player_angle += self.player_angular_velocity
        player_pos = self._get_player_pos()

        new_dist = self._dist_to_target()
        reward += 0.1 if new_dist < old_dist else -0.1
            
        self.shrink_timer -= 1
        if self.shrink_timer <= 0:
            self.player_size *= 0.8
            self.shrink_timer = self.shrink_interval
            if self.player_size < 10:
                self._create_particles(player_pos, (255,255,255), 10, 1)

        self._update_particles()
        self._update_stars()

        # --- 3. Check Collisions & Events ---
        target_gate_id = self.target_gate_order[self.current_target_idx]
        gate_orbit_radius = self.SCREEN_HEIGHT / 2 - 30

        if abs(self.player_orbit_radius - gate_orbit_radius) < self.player_size:
            player_angle_norm = self.player_angle % (2 * math.pi)
            for i, gate in enumerate(self.gates):
                half_angular_width = math.atan2(gate["width"] / 2, gate_orbit_radius)
                angle_diff = abs((player_angle_norm - gate["angle"] + math.pi) % (2 * math.pi) - math.pi)

                if angle_diff < half_angular_width:
                    if i == target_gate_id and self.player_color_index == gate["color_index"]:
                        self.score += 10
                        reward += 10
                        self.current_target_idx += 1
                        self._create_particles(player_pos, self.COLOR_PALETTE[gate["color_index"]], 100, 5)
                        self.player_size = min(35.0, self.player_size * 1.1) 
                    else:
                        self.score -= 5
                        reward -= 5.0
                        self.game_over = True
                        reward -= 100.0
                        self._create_particles(player_pos, (200,200,200), 200, 2, is_explosion=True)
                    break

        # --- 4. Check Termination Conditions ---
        if self.game_over:
            terminated = True
        elif self.current_target_idx >= len(self.gates):
            self.win = True
            self.game_over = True
            terminated = True
            self.score += 100
            reward += 100.0
        elif self.player_size < 1.0:
            self.game_over = True
            terminated = True
            self.score -= 100
            reward -= 100.0
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        
        self.score = max(-100, self.score)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_player_pos(self):
        x = self.CENTER[0] + self.player_orbit_radius * math.cos(self.player_angle)
        y = self.CENTER[1] + self.player_orbit_radius * math.sin(self.player_angle)
        return (x, y)

    def _dist_to_target(self):
        player_pos = self._get_player_pos()
        target_gate = self.gates[self.target_gate_order[self.current_target_idx]]
        return math.hypot(player_pos[0] - target_gate["pos"][0], player_pos[1] - target_gate["pos"][1])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "gates_passed": self.current_target_idx}

    def _render_game(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star["color"], (int(star["x"]), int(star["y"])), int(star["size"]))

        target_gate_id = self.target_gate_order[self.current_target_idx]
        for i, gate in enumerate(self.gates):
            color = self.COLOR_PALETTE[gate["color_index"]]
            is_target = (i == target_gate_id) and not self.game_over
            
            gate_thickness = 15 if is_target else 8
            if is_target:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                gate_thickness += int(pulse * 5)
            
            rect = pygame.Rect(0, 0, gate["width"], gate_thickness)
            rect.center = gate["pos"]
            
            angle_deg = math.degrees(gate["angle"]) + 90
            s = pygame.Surface(rect.size, pygame.SRCALPHA)
            s.fill(color)
            rotated_s = pygame.transform.rotate(s, -angle_deg)
            new_rect = rotated_s.get_rect(center=rect.center)
            self.screen.blit(rotated_s, new_rect)

        player_pos = self._get_player_pos()
        player_color = self.COLOR_PALETTE[self.player_color_index]
        
        glow_size = int(self.player_size * 1.8)
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*player_color, 60), (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surf, (int(player_pos[0] - glow_size), int(player_pos[1] - glow_size)))

        pygame.draw.circle(self.screen, player_color, (int(player_pos[0]), int(player_pos[1])), int(self.player_size))
        pygame.draw.circle(self.screen, self.COLOR_WHITE, (int(player_pos[0]), int(player_pos[1])), int(self.player_size), 2)

        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], (int(p["x"]), int(p["y"])), int(p["size"]))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

        if not self.game_over:
            target_gate_id = self.target_gate_order[self.current_target_idx]
            target_color = self.COLOR_PALETTE[self.gates[target_gate_id]["color_index"]]
            pygame.draw.rect(self.screen, target_color, (self.SCREEN_WIDTH - 50, 10, 40, 40))
            pygame.draw.rect(self.screen, self.COLOR_WHITE, (self.SCREEN_WIDTH - 50, 10, 40, 40), 2)

        player_pos = self._get_player_pos()
        bar_width = 80
        bar_height = 8
        bar_x = player_pos[0] - bar_width / 2
        bar_y = player_pos[1] - self.player_size - 20
        
        fill_ratio = self.shrink_timer / self.shrink_interval
        fill_width = bar_width * fill_ratio
        
        bar_color = (80, 200, 80) if fill_ratio > 0.5 else (220, 220, 80) if fill_ratio > 0.2 else (200, 80, 80)
        
        pygame.draw.rect(self.screen, (50, 50, 50), (int(bar_x), int(bar_y), bar_width, bar_height))
        pygame.draw.rect(self.screen, bar_color, (int(bar_x), int(bar_y), int(fill_width), bar_height))

        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PALETTE[1] if self.win else self.COLOR_PALETTE[0]
            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=self.CENTER)
            self.screen.blit(end_text, text_rect)

    def _create_stars(self):
        self.stars.clear()
        for _ in range(100):
            self.stars.append({
                "x": self.np_random.uniform(0, self.SCREEN_WIDTH),
                "y": self.np_random.uniform(0, self.SCREEN_HEIGHT),
                "size": self.np_random.uniform(0.5, 2),
                "speed": self.np_random.uniform(0.1, 0.5),
                "color": (
                    self.np_random.integers(50, 101),
                    self.np_random.integers(50, 101),
                    self.np_random.integers(70, 121)
                )
            })

    def _update_stars(self):
        for star in self.stars:
            star["x"] -= star["speed"]
            if star["x"] < 0:
                star["x"] = self.SCREEN_WIDTH
                star["y"] = self.np_random.uniform(0, self.SCREEN_HEIGHT)

    def _create_particles(self, pos, color, count, max_speed, is_explosion=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            self.particles.append({
                "x": pos[0], "y": pos[1],
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "size": self.np_random.uniform(2, 5),
                "life": self.np_random.integers(20, 40),
                "color": color,
                "explosion": is_explosion
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            if not p["explosion"]:
                p["vy"] += 0.1 # Gravity
            p["life"] -= 1
            p["size"] *= 0.97
            if p["life"] <= 0 or p["size"] < 0.5:
                self.particles.remove(p)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # Unset the dummy video driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    movement_action = 0
    space_action = 0
    shift_action = 0

    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Color Gate Navigator")
    clock = pygame.time.Clock()

    print(GameEnv.user_guide)
    print("R: Reset, Q: Quit")
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key == pygame.K_SPACE:
                    space_action = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_action = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    space_action = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_action = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        else:
            movement_action = 0

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode Finished. Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(env.FPS)
    
    env.close()