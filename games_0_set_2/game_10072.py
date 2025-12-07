import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:45:17.968786
# Source Brief: brief_00072.md
# Brief Index: 72
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a shape-shifting entity.
    The goal is to pass through gates that match the entity's current form
    to score points, aiming to reach 100 points within 60 seconds.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]`: Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - `actions[1]`: Form Shift (0: released, 1: pressed) - cycles form on press.
    - `actions[2]`: Unused (0: released, 1: pressed)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - Pass through matching gate: +1 (Line), +2 (Square), +4 (Pyramid)
    - Collide with mismatched gate: -1
    - Move towards the next gate: +0.1
    - Move away from the next gate: -0.1
    - Win the game (score >= 100): +100

    **Termination:**
    - Score reaches 100.
    - Timer of 60 seconds (1800 steps) runs out.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Control a shape-shifting entity and pass through gates that match your current form "
        "to score points before the timer runs out."
    )
    user_guide = "Controls: Use arrow keys or WASD to move. Press space to cycle through shapes."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_DURATION_SECONDS = 60
    WIN_SCORE = 100

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 60)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_SHADOW = (10, 10, 15)

    # Form-specific properties
    FORM_PROPS = {
        0: {"name": "Line", "color": (50, 150, 255), "glow": (20, 75, 128), "score": 5, "reward": 1},
        1: {"name": "Square", "color": (50, 255, 150), "glow": (20, 128, 75), "score": 10, "reward": 2},
        2: {"name": "Pyramid", "color": (255, 100, 100), "glow": (128, 50, 50), "score": 20, "reward": 4},
    }
    COLLISION_PENALTY_SCORE = -5
    COLLISION_PENALTY_REWARD = -1

    # Player settings
    PLAYER_SPEED = 12
    PLAYER_LERP_RATE = 0.25

    # Gate settings
    GATE_SPAWN_INTERVAL = 30
    GATE_SPEED = 1.5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        try:
            self.font_large = pygame.font.Font(None, 48)
        except pygame.error:
            self.font_large = pygame.font.SysFont("monospace", 42)

        # State variables are initialized in reset()
        self.steps = 0
        self.max_steps = self.GAME_DURATION_SECONDS * self.metadata["render_fps"]
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_target_pos = [0, 0]
        self.player_form = 0
        self.player_size = 20
        self.gates = []
        self.particles = []
        self.prev_space_held = False
        self.time_since_last_gate = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.75]
        self.player_target_pos = self.player_pos.copy()
        self.player_form = 0
        self.gates = []
        self.particles = []
        self.time_since_last_gate = 0
        self.prev_space_held = False

        for _ in range(4):
            self._spawn_gate(initial=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        prev_dist_to_gate = self._get_dist_to_next_gate()

        self._update_player(movement, space_held)
        self._update_gates()
        self._update_particles()
        
        self.player_pos[0] += (self.player_target_pos[0] - self.player_pos[0]) * self.PLAYER_LERP_RATE
        self.player_pos[1] += (self.player_target_pos[1] - self.player_pos[1]) * self.PLAYER_LERP_RATE
        
        event_reward = self._check_collisions_and_scoring()
        
        new_dist_to_gate = self._get_dist_to_next_gate()
        movement_reward = 0
        if prev_dist_to_gate is not None and new_dist_to_gate is not None:
            if new_dist_to_gate < prev_dist_to_gate:
                movement_reward = 0.1
            else:
                movement_reward = -0.1
        
        reward = event_reward + movement_reward
        
        self.steps += 1
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
            self._create_particles(self.player_pos, (255, 255, 100), 100, 5.0)
        elif self.steps >= self.max_steps:
            terminated = True # In new API, this is often a truncation
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, movement, space_held):
        if movement == 1: self.player_target_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: self.player_target_pos[1] += self.PLAYER_SPEED
        elif movement == 3: self.player_target_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: self.player_target_pos[0] += self.PLAYER_SPEED
            
        self.player_target_pos[0] = np.clip(self.player_target_pos[0], self.player_size, self.SCREEN_WIDTH - self.player_size)
        self.player_target_pos[1] = np.clip(self.player_target_pos[1], self.player_size, self.SCREEN_HEIGHT - self.player_size)

        if space_held and not self.prev_space_held:
            self.player_form = (self.player_form + 1) % len(self.FORM_PROPS)
            # SFX: Play form shift sound
            self._create_particles(self.player_pos, self.FORM_PROPS[self.player_form]["color"], 15, 2.0)
        self.prev_space_held = space_held

    def _update_gates(self):
        for gate in self.gates:
            gate['pos'][1] += self.GATE_SPEED
        self.gates = [g for g in self.gates if g['pos'][1] < self.SCREEN_HEIGHT + 50]
        self.time_since_last_gate += 1
        if self.time_since_last_gate >= self.GATE_SPAWN_INTERVAL:
            self._spawn_gate()
            self.time_since_last_gate = 0

    def _spawn_gate(self, initial=False):
        form = self.np_random.integers(0, len(self.FORM_PROPS))
        x = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
        y_range = (50, self.SCREEN_HEIGHT - 100) if initial else (-100, -50)
        y = self.np_random.uniform(*y_range)
        self.gates.append({
            'pos': [x, y], 'form': form, 'size': self.player_size * 1.5
        })

    def _check_collisions_and_scoring(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - self.player_size, self.player_pos[1] - self.player_size, self.player_size*2, self.player_size*2)
        collided_gate_idx = -1
        for i, gate in enumerate(self.gates):
            gate_rect = pygame.Rect(gate['pos'][0] - gate['size'], gate['pos'][1] - gate['size'], gate['size']*2, gate['size']*2)
            if player_rect.colliderect(gate_rect):
                if self.player_form == gate['form']:
                    props = self.FORM_PROPS[gate['form']]
                    self.score += props["score"]
                    reward += props["reward"]
                    # SFX: Play success chime
                    self._create_particles(gate['pos'], props["color"], 50, 3.0)
                else:
                    self.score = max(0, self.score + self.COLLISION_PENALTY_SCORE)
                    reward += self.COLLISION_PENALTY_REWARD
                    # SFX: Play collision/error sound
                    self._create_particles(self.player_pos, (255, 50, 50), 20, 2.5)
                collided_gate_idx = i
                break
        if collided_gate_idx != -1:
            del self.gates[collided_gate_idx]
        return reward

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': self.np_random.integers(20, 40)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_gates()
        self._render_particles()
        self._render_player()

    def _render_grid(self):
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i), 1)

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        size = self.player_size
        props = self.FORM_PROPS[self.player_form]
        color, glow_color = props["color"], props["glow"]

        glow_radius = int(size * 1.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, glow_color + (60,), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        if self.player_form == 0: # Line
            pygame.draw.rect(self.screen, color, (x - size * 1.5, y - size * 0.25, size * 3, size * 0.5))
        elif self.player_form == 1: # Square
            pygame.draw.rect(self.screen, color, (x - size, y - size, size * 2, size * 2))
        elif self.player_form == 2: # Pyramid
            points = [(x, y - size * 1.2), (x - size, y + size * 0.8), (x + size, y + size * 0.8)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_gates(self):
        for gate in self.gates:
            x, y = int(gate['pos'][0]), int(gate['pos'][1])
            size = int(gate['size'])
            color = self.FORM_PROPS[gate['form']]["color"]
            if gate['form'] == 0: # Line
                pygame.draw.rect(self.screen, color, (x - size * 1.5, y - size * 0.25, size * 3, size * 0.5), 3)
            elif gate['form'] == 1: # Square
                pygame.draw.rect(self.screen, color, (x - size, y - size, size * 2, size * 2), 3)
            elif gate['form'] == 2: # Pyramid
                points = [(x, y - size * 1.2), (x - size, y + size * 0.8), (x + size, y + size * 0.8)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = max(1, int(p['life'] / 8))
            pygame.draw.circle(self.screen, p['color'], pos, size)

    def _render_ui(self):
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        score_shadow = self.font_large.render(f"{self.score}", True, self.COLOR_UI_SHADOW)
        self.screen.blit(score_shadow, (12, 12))
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, self.GAME_DURATION_SECONDS - (self.steps / self.metadata["render_fps"]))
        timer_str = f"{int(time_left // 60):02}:{int(time_left % 60):02}"
        timer_color = (255, 100, 100) if time_left < 10 else self.COLOR_UI_TEXT
        timer_text = self.font_large.render(timer_str, True, timer_color)
        timer_shadow = self.font_large.render(timer_str, True, self.COLOR_UI_SHADOW)
        self.screen.blit(timer_shadow, timer_shadow.get_rect(topright=(self.SCREEN_WIDTH - 8, 12)))
        self.screen.blit(timer_text, timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_dist_to_next_gate(self):
        if not self.gates: return None
        player_x, player_y = self.player_target_pos
        # Prioritize gates that are in front of the player
        upcoming_gates = [g for g in self.gates if g['pos'][1] > self.player_pos[1] - self.player_size * 2]
        if not upcoming_gates:
            # If no gates are in front, target the overall closest one
            target_gate = min(self.gates, key=lambda g: math.hypot(g['pos'][0] - player_x, g['pos'][1] - player_y))
        else:
            # Otherwise, target the closest gate in terms of y-distance that is in front
            target_gate = min(upcoming_gates, key=lambda g: g['pos'][1])
        return math.hypot(target_gate['pos'][0] - player_x, target_gate['pos'][1] - player_y)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # The __main__ block is for human play and is not used by the evaluation system.
    # It will be ignored, so you don't need to fix it.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    env = GameEnv(render_mode="rgb_array")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Shape Shifter")
    obs, info = env.reset()
    terminated = False
    
    running = True
    while running:
        movement, space_held, shift_held = 0, 0, 0
        
        # Poll for events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Check for key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}. Press 'R' to restart.")
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                        truncated = False
                        wait_for_reset = False
        
        env.clock.tick(env.metadata["render_fps"])

    env.close()