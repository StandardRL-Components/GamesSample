import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import copy
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Gymnasium environment for a steampunk puzzle-platformer.
    The agent controls a robot exploring a clockwork colossus, using portals
    and a time-rewind mechanic to activate colored circuits.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a robot in a steampunk colossus, using portals and a time-rewind mechanic "
        "to activate a series of colored circuits before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to rewind time and shift to place or use a teleportation portal."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 500
    PLAYER_SPEED = 5.0
    PLAYER_RADIUS = 10
    CIRCUIT_RADIUS = 15
    INTERACTION_DISTANCE = PLAYER_RADIUS + CIRCUIT_RADIUS
    NUM_CIRCUITS = 5
    REWIND_STEPS = 10

    # --- COLORS ---
    COLOR_BG = (20, 30, 40)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_PORTAL = (255, 128, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_PROGRESS_BAR_BG = (50, 50, 70)
    COLOR_PROGRESS_BAR_FG = (100, 180, 255)
    CIRCUIT_COLORS = {
        "red": (255, 50, 50),
        "green": (50, 255, 50),
        "blue": (50, 100, 255),
    }
    CIRCUIT_COLORS_UNMATCHED = {
        "red": (80, 20, 20),
        "green": (20, 80, 20),
        "blue": (20, 40, 80),
    }
    GEAR_COLORS = [(40, 50, 60), (45, 55, 65), (50, 60, 70)]
    PIPE_COLOR = (60, 70, 80)


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
        self.font_msg = pygame.font.SysFont("monospace", 36, bold=True)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.circuits = []
        self.portal_pos = np.array([0.0, 0.0])
        self.portal_active = False
        self.history = []
        self.background_gears = []
        self.background_pipes = []
        
        # Action state tracking for rising edge detection
        self.space_was_held = False
        self.shift_was_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        
        self.portal_active = False
        self.portal_pos = np.array([0.0, 0.0])

        self.space_was_held = False
        self.shift_was_held = False
        
        self._generate_level()

        # History for time rewind
        self.history = []
        self._save_state_to_history()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # --- Action Processing ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # Rising edge detection for discrete actions
        space_pressed = space_held and not self.space_was_held
        shift_pressed = shift_held and not self.shift_was_held
        self.space_was_held = space_held
        self.shift_was_held = shift_held
        
        # --- Game Logic ---
        prev_player_pos = self.player_pos.copy()

        # 1. Player Movement
        self._handle_movement(movement)
        
        # 2. Time Rewind (Space)
        if space_pressed:
            self._rewind_time()
            reward -= 1 # Small penalty for rewinding
            # Sound: play "rewind" sound

        # 3. Portal (Shift)
        if shift_pressed:
            self._handle_portal()
            # Sound: play "portal_open" or "teleport" sound

        # 4. Circuit Interaction
        reward += self._check_circuit_interaction()
        
        # 5. Continuous Reward
        reward += self._calculate_distance_reward(prev_player_pos)

        # 6. Update state
        self.steps += 1
        self.score += reward
        self._save_state_to_history()

        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            num_matched = sum(1 for c in self.circuits if c['matched'])
            if num_matched == self.NUM_CIRCUITS:
                reward += 100 # Victory bonus
                # Sound: play "victory_fanfare"
            else:
                reward -= 100 # Timeout penalty
                # Sound: play "failure_buzzer"
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_circuits()
        if self.portal_active:
            self._render_portal()
        self._render_player()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_message()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "circuits_matched": sum(1 for c in self.circuits if c['matched']),
            "portal_active": self.portal_active,
        }

    def close(self):
        pygame.quit()

    # --- Private Helper Methods: Game Logic ---

    def _generate_level(self):
        # Background elements
        self.background_gears = []
        for _ in range(15):
            self.background_gears.append({
                'pos': (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                'radius': self.np_random.integers(20, 100),
                'speed': self.np_random.uniform(0.1, 0.5) * self.np_random.choice([-1, 1]),
                'color': random.choice(self.GEAR_COLORS),
                'teeth': self.np_random.integers(8, 20)
            })
        
        self.background_pipes = []
        for _ in range(10):
            self.background_pipes.append({
                'start': (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                'end': (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                'width': self.np_random.integers(10, 20)
            })

        # Interactive elements (Circuits)
        self.circuits = []
        color_names = list(self.CIRCUIT_COLORS.keys())
        for i in range(self.NUM_CIRCUITS):
            placed = False
            while not placed:
                pos = np.array([
                    self.np_random.uniform(self.CIRCUIT_RADIUS, self.SCREEN_WIDTH - self.CIRCUIT_RADIUS),
                    self.np_random.uniform(self.CIRCUIT_RADIUS, self.SCREEN_HEIGHT - self.CIRCUIT_RADIUS)
                ])
                # Ensure no overlap with other circuits
                if all(np.linalg.norm(pos - c['pos']) > self.CIRCUIT_RADIUS * 3 for c in self.circuits):
                    self.circuits.append({
                        'pos': pos,
                        'color_name': color_names[i % len(color_names)],
                        'matched': False
                    })
                    placed = True
    
    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

    def _handle_portal(self):
        if self.portal_active:
            # Teleport to portal and deactivate it
            self.player_pos = self.portal_pos.copy()
            self.portal_active = False
        else:
            # Create portal at current position
            self.portal_pos = self.player_pos.copy()
            self.portal_active = True
            
    def _rewind_time(self):
        if len(self.history) > self.REWIND_STEPS:
            # Pop current state and rewind
            self.history.pop() 
            for _ in range(self.REWIND_STEPS):
                if len(self.history) > 1:
                    self.history.pop()
            
            # Restore the state
            last_state = self.history[-1]
            self.steps = last_state['steps']
            self.score = last_state['score']
            self.player_pos = last_state['player_pos'].copy()
            self.circuits = copy.deepcopy(last_state['circuits'])
            self.portal_active = last_state['portal_active']
            self.portal_pos = last_state['portal_pos'].copy()
    
    def _check_circuit_interaction(self):
        reward = 0
        for circuit in self.circuits:
            if not circuit['matched']:
                distance = np.linalg.norm(self.player_pos - circuit['pos'])
                if distance < self.INTERACTION_DISTANCE:
                    circuit['matched'] = True
                    reward += 5
                    # Sound: play "circuit_matched" sound
        return reward

    def _calculate_distance_reward(self, prev_pos):
        unmatched_circuits = [c for c in self.circuits if not c['matched']]
        if not unmatched_circuits:
            return 0

        # Find distance to closest unmatched circuit
        min_dist_before = min(np.linalg.norm(prev_pos - c['pos']) for c in unmatched_circuits)
        min_dist_after = min(np.linalg.norm(self.player_pos - c['pos']) for c in unmatched_circuits)
        
        # Reward for getting closer, scaled down
        return (min_dist_before - min_dist_after) * 0.01

    def _check_termination(self):
        all_matched = all(c['matched'] for c in self.circuits)
        time_out = self.steps >= self.MAX_STEPS
        return all_matched or time_out

    def _save_state_to_history(self):
        state = {
            'steps': self.steps,
            'score': self.score,
            'player_pos': self.player_pos.copy(),
            'circuits': copy.deepcopy(self.circuits),
            'portal_active': self.portal_active,
            'portal_pos': self.portal_pos.copy()
        }
        self.history.append(state)
        # Keep history buffer from growing indefinitely
        if len(self.history) > self.MAX_STEPS + 5:
            self.history.pop(0)

    # --- Private Helper Methods: Rendering ---

    def _render_background(self):
        # Render Pipes
        for pipe in self.background_pipes:
            pygame.draw.line(self.screen, self.PIPE_COLOR, pipe['start'], pipe['end'], pipe['width'])
        
        # Render Gears
        for gear in self.background_gears:
            x, y = int(gear['pos'][0]), int(gear['pos'][1])
            radius = int(gear['radius'])
            angle = (self.steps * gear['speed']) % 360
            
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, gear['color'])
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, tuple(min(255, c+10) for c in gear['color']))
            
            # Draw teeth/spokes for detail
            for i in range(gear['teeth']):
                a = math.radians(angle + i * (360 / gear['teeth']))
                sx = x + math.cos(a) * (radius - 5)
                sy = y + math.sin(a) * (radius - 5)
                ex = x + math.cos(a) * (radius + 5)
                ey = y + math.sin(a) * (radius + 5)
                pygame.draw.line(self.screen, gear['color'], (sx, sy), (ex, ey), 5)

    def _render_circuits(self):
        for circuit in self.circuits:
            x, y = int(circuit['pos'][0]), int(circuit['pos'][1])
            color_name = circuit['color_name']
            
            if circuit['matched']:
                color = self.CIRCUIT_COLORS[color_name]
                # Glowing effect for matched circuits
                for i in range(5):
                    glow_color = (*color, int(150 / (i + 1)))
                    pygame.gfxdraw.filled_circle(self.screen, x, y, self.CIRCUIT_RADIUS + i * 2, glow_color)
                pygame.gfxdraw.filled_circle(self.screen, x, y, self.CIRCUIT_RADIUS, color)
                pygame.gfxdraw.aacircle(self.screen, x, y, self.CIRCUIT_RADIUS, (255, 255, 255))
            else:
                unmatched_color = self.CIRCUIT_COLORS_UNMATCHED[color_name]
                color = self.CIRCUIT_COLORS[color_name]
                pygame.gfxdraw.filled_circle(self.screen, x, y, self.CIRCUIT_RADIUS, (30,30,30))
                pygame.gfxdraw.filled_circle(self.screen, x, y, self.CIRCUIT_RADIUS - 4, unmatched_color)
                pygame.gfxdraw.aacircle(self.screen, x, y, self.CIRCUIT_RADIUS, color)

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        
        # Glow effect
        glow_radius = int(self.PLAYER_RADIUS * 1.8)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (x - glow_radius, y - glow_radius))

        # Player body
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.PLAYER_RADIUS, self.COLOR_BG)

    def _render_portal(self):
        x, y = int(self.portal_pos[0]), int(self.portal_pos[1])
        for i in range(4):
            # Animate radius and alpha for a swirling effect
            offset = self.steps * 0.1 + i * (math.pi / 2)
            radius = int(20 + 8 * math.sin(offset))
            alpha = int(100 + 90 * math.cos(offset * 0.7))
            alpha = max(0, min(255, alpha))
            
            if radius > 0:
                color = (*self.COLOR_PORTAL, alpha)
                pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
                pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _render_ui(self):
        # Time remaining
        time_text = self.font_ui.render(f"TIME: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Progress bar
        num_matched = sum(1 for c in self.circuits if c['matched'])
        progress = num_matched / self.NUM_CIRCUITS
        
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_FG, (10, 10, int(bar_width * progress), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, 10, bar_width, bar_height), 1)

        progress_text = self.font_ui.render(f"{num_matched}/{self.NUM_CIRCUITS} CIRCUITS", True, self.COLOR_UI_TEXT)
        self.screen.blit(progress_text, (15, 12))

    def _render_game_over_message(self):
        num_matched = sum(1 for c in self.circuits if c['matched'])
        if num_matched == self.NUM_CIRCUITS:
            msg = "COLOSSUS RESTORED"
            color = (150, 255, 150)
        else:
            msg = "OUT OF TIME"
            color = (255, 150, 150)
        
        text_surf = self.font_msg.render(msg, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        # Add a dark background for readability
        bg_rect = text_rect.inflate(20, 20)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 150))
        self.screen.blit(bg_surf, bg_rect)

        self.screen.blit(text_surf, text_rect)


# --- Example Usage ---
if __name__ == "__main__":
    # The main execution block is for human play and requires a display.
    # We will unset the dummy video driver to allow a window to be created.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Clockwork Colossus")
    game_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while not done:
        # Action mapping for human player
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the game window
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        clock.tick(30) # Limit to 30 FPS for human play

    env.close()