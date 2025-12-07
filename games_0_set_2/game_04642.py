
# Generated: 2025-08-28T03:00:56.586676
# Source Brief: brief_04642.md
# Brief Index: 4642

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select/connect points. Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Repair a robot by connecting circuits on an isometric grid before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_CHASSIS = (40, 50, 70)
    COLOR_POINT = (100, 120, 180)
    COLOR_POINT_LIT = (220, 240, 255)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_WIRE_ACTIVE = (0, 150, 255)
    COLOR_WIRE_CORRECT = (0, 255, 100)
    COLOR_WIRE_INCORRECT = (255, 50, 50)
    COLOR_LED_OFF = (60, 60, 60)
    COLOR_LED_ON = (255, 220, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_WARN = (255, 100, 100)
    COLOR_TEXT_WIN = (100, 255, 100)

    # Grid and Isometric Projection
    GRID_SIZE = (12, 8)
    TILE_WIDTH = 40
    TILE_HEIGHT = TILE_WIDTH / 2
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 120

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.points = []
        self.solutions = []
        self.all_solution_connections = set()
        self.player_connections = set()
        self.circuits_completed = []
        self.selecting_from_point_idx = None
        self.particles = []
        self.prev_action = np.array([0, 0, 0])
        
        # Initialize state
        self.reset()
        
        # Self-check
        self.validate_implementation()

    def _project_iso(self, gx, gy):
        """Converts grid coordinates to screen coordinates."""
        screen_x = self.ORIGIN_X + (gx - gy) * self.TILE_WIDTH / 2
        screen_y = self.ORIGIN_Y + (gx + gy) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _define_puzzle(self):
        """Creates the connection points and circuit solutions."""
        self.points = [
            (1, 1), (3, 1), (5, 1), (7, 1), (9, 1),  # Row 1: 0-4
            (1, 3), (3, 3), (5, 3), (7, 3), (9, 3),  # Row 2: 5-9
            (1, 5), (3, 5), (5, 5), (7, 5), (9, 5),  # Row 3: 10-14
        ]
        
        # Solutions are sets of sorted tuples of point indices
        self.solutions = [
            { (0, 5), (5, 6), (1, 6) },                                   # Circuit 1 (Z-shape)
            { (2, 7), (7, 12), (11, 12) },                                # Circuit 2 (U-shape)
            { (3, 4), (4, 9), (8, 9) },                                   # Circuit 3 (Stairs)
            { (10, 11) },                                                 # Circuit 4 (Simple line)
            { (13, 14), (8, 13) }                                         # Circuit 5 (L-shape)
        ]
        
        self.all_solution_connections = set()
        for circuit in self.solutions:
            self.all_solution_connections.update(circuit)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        self._define_puzzle()
        
        self.player_connections = set()
        self.circuits_completed = [False] * len(self.solutions)
        self.selecting_from_point_idx = None
        self.particles = []
        self.prev_action = np.array([0, 0, 0])

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- Handle Input and State Changes ---
        movement, space_pressed, shift_pressed = self._process_input(action)
        self._handle_movement(movement)
        
        if space_pressed:
            reward += self._handle_selection()
        
        if shift_pressed:
            self._handle_deselection()

        # --- Update Game Logic ---
        self._update_particles()
        
        # Check for new circuit completions
        newly_completed = self._check_circuit_completion()
        if newly_completed:
            reward += 1.0 * newly_completed
            # sfx: circuit_complete_sound

        # --- Check Termination Conditions ---
        time_up = self.steps >= self.MAX_STEPS
        all_circuits_done = all(self.circuits_completed)
        
        if time_up or all_circuits_done:
            terminated = True
            self.game_over = True
            if all_circuits_done:
                reward += 100  # Win bonus
                # sfx: win_jingle
            else:
                reward -= 100  # Timeout penalty
                # sfx: lose_buzzer

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _process_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # Detect rising edge for presses
        space_pressed = space_held and not self.prev_action[1]
        shift_pressed = shift_held and not self.prev_action[2]
        
        self.prev_action = [movement, space_held, shift_held]
        
        return movement, space_pressed, shift_pressed

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE[0] - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE[1] - 1)

    def _handle_selection(self):
        """Handles the 'space' action to select or connect points."""
        reward = 0
        target_idx = self._get_point_at_cursor()
        
        if target_idx is not None:
            if self.selecting_from_point_idx is None:
                # Start a new wire
                self.selecting_from_point_idx = target_idx
                # sfx: select_point_sound
            elif self.selecting_from_point_idx != target_idx:
                # Complete a wire connection
                p1 = min(self.selecting_from_point_idx, target_idx)
                p2 = max(self.selecting_from_point_idx, target_idx)
                
                # Avoid duplicate connections
                if not any(conn[1] == p1 and conn[2] == p2 for conn in self.player_connections):
                    is_correct = (p1, p2) in self.all_solution_connections
                    connection_type = 'correct' if is_correct else 'incorrect'
                    self.player_connections.add((connection_type, p1, p2))
                    
                    if is_correct:
                        reward = 0.1
                        self._create_particles(self.points[p1], self.points[p2], self.COLOR_WIRE_CORRECT)
                        # sfx: correct_connect_sound
                    else:
                        reward = -0.1
                        self._create_particles(self.points[p1], self.points[p2], self.COLOR_WIRE_INCORRECT)
                        # sfx: incorrect_connect_sound
                
                self.selecting_from_point_idx = None
        return reward

    def _handle_deselection(self):
        """Handles the 'shift' action to cancel a wire selection."""
        if self.selecting_from_point_idx is not None:
            self.selecting_from_point_idx = None
            # sfx: deselect_sound

    def _get_point_at_cursor(self):
        """Returns the index of the connection point at the cursor, or None."""
        for i, point_pos in enumerate(self.points):
            if point_pos[0] == self.cursor_pos[0] and point_pos[1] == self.cursor_pos[1]:
                return i
        return None

    def _check_circuit_completion(self):
        """Checks if any circuits were newly completed and updates their status."""
        newly_completed_count = 0
        player_correct_connections = { (p1, p2) for type, p1, p2 in self.player_connections if type == 'correct' }
        
        for i, solution_set in enumerate(self.solutions):
            if not self.circuits_completed[i]:
                if solution_set.issubset(player_correct_connections):
                    self.circuits_completed[i] = True
                    newly_completed_count += 1
        return newly_completed_count

    def _get_observation(self):
        # Main drawing function
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders all game world elements."""
        # Draw chassis background
        chassis_rect_iso = [self._project_iso(0, 0), self._project_iso(self.GRID_SIZE[0], 0),
                            self._project_iso(self.GRID_SIZE[0], self.GRID_SIZE[1]), self._project_iso(0, self.GRID_SIZE[1])]
        pygame.gfxdraw.filled_polygon(self.screen, chassis_rect_iso, self.COLOR_CHASSIS)
        pygame.gfxdraw.aapolygon(self.screen, chassis_rect_iso, self.COLOR_POINT)

        # Draw player connections
        for type, p1_idx, p2_idx in self.player_connections:
            pos1 = self._project_iso(*self.points[p1_idx])
            pos2 = self._project_iso(*self.points[p2_idx])
            color = self.COLOR_WIRE_CORRECT if type == 'correct' else self.COLOR_WIRE_INCORRECT
            pygame.draw.line(self.screen, color, pos1, pos2, 3)

        # Draw active wire being selected
        if self.selecting_from_point_idx is not None:
            start_pos = self._project_iso(*self.points[self.selecting_from_point_idx])
            cursor_screen_pos = self._project_iso(*self.cursor_pos)
            # Pulsating effect for active wire
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            color = (
                self.COLOR_WIRE_ACTIVE[0],
                int(self.COLOR_WIRE_ACTIVE[1] * (0.5 + pulse * 0.5)),
                self.COLOR_WIRE_ACTIVE[2]
            )
            pygame.draw.aaline(self.screen, color, start_pos, cursor_screen_pos, 2)
        
        # Draw connection points
        for i, point_pos in enumerate(self.points):
            screen_pos = self._project_iso(*point_pos)
            is_selected = (self.selecting_from_point_idx == i)
            radius = 6 if is_selected else 4
            color = self.COLOR_POINT_LIT if is_selected else self.COLOR_POINT
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius, color)

        # Draw cursor
        cursor_screen_pos = self._project_iso(*self.cursor_pos)
        size = 12 + math.sin(self.steps * 0.2) * 2
        pts = [
            (cursor_screen_pos[0], cursor_screen_pos[1] - size),
            (cursor_screen_pos[0] + size, cursor_screen_pos[1]),
            (cursor_screen_pos[0], cursor_screen_pos[1] + size),
            (cursor_screen_pos[0] - size, cursor_screen_pos[1]),
        ]
        pygame.draw.aalines(self.screen, self.COLOR_CURSOR, True, pts, 2)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

    def _render_ui(self):
        """Renders UI elements like score, timer, and LEDs."""
        # Timer
        remaining_seconds = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = f"{max(0, remaining_seconds):.1f}"
        timer_color = self.COLOR_TEXT_WARN if remaining_seconds < 10 else self.COLOR_TEXT
        text_surf = self.font_timer.render(timer_text, True, timer_color)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 20, 10))
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        text_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (20, 10))

        # Circuit LEDs
        led_y = 50
        for i, completed in enumerate(self.circuits_completed):
            color = self.COLOR_LED_ON if completed else self.COLOR_LED_OFF
            pos = (30, led_y + i * 25)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, color)
            # Glow effect
            if completed:
                glow_color = (*self.COLOR_LED_ON[:3], 50)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, glow_color)


        # Game Over / Win Message
        if self.game_over:
            if all(self.circuits_completed):
                msg = "SYSTEM REPAIRED"
                color = self.COLOR_TEXT_WIN
            else:
                msg = "TIME OUT"
                color = self.COLOR_TEXT_WARN
            
            text_surf = self.font_msg.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            # Draw a semi-transparent background for the message
            bg_rect = text_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 150))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "circuits_completed": sum(self.circuits_completed),
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _create_particles(self, grid_pos1, grid_pos2, color):
        """Creates a burst of particles along a line."""
        start_pos = self._project_iso(*grid_pos1)
        end_pos = self._project_iso(*grid_pos2)
        
        for i in range(20): # 20 particles
            t = i / 19.0
            pos = [start_pos[0] * (1 - t) + end_pos[0] * t,
                   start_pos[1] * (1 - t) + end_pos[1] * t]
            
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            
            self.particles.append({
                'pos': pos,
                'vel': vel,
                'radius': random.uniform(1, 3),
                'life': random.randint(15, 30), # frames
                'color': color
            })

    def _update_particles(self):
        """Updates position, life, and removes dead particles."""
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] *= 0.95
            if p['life'] > 0 and p['radius'] > 0.5:
                active_particles.append(p)
        self.particles = active_particles
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# This part is for standalone testing and visualization
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for display
    pygame.display.set_caption("Circuit Repair")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Map keyboard keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP:    (1, 0, 0),
        pygame.K_DOWN:  (2, 0, 0),
        pygame.K_LEFT:  (3, 0, 0),
        pygame.K_RIGHT: (4, 0, 0),
    }

    while running:
        # --- Human Input ---
        # Default action is no-op
        action = np.array([0, 0, 0])
        
        keys_pressed = pygame.key.get_pressed()
        
        # Movement
        for key, (move_val, _, _) in key_map.items():
            if keys_pressed[key]:
                action[0] = move_val
                break # Prioritize one movement key
                
        # Space and Shift
        if keys_pressed[pygame.K_SPACE]:
            action[1] = 1
        if keys_pressed[pygame.K_LSHIFT] or keys_pressed[pygame.K_RSHIFT]:
            action[2] = 1
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            print("Press 'R' to reset.")
            # Wait for reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("Resetting environment.")
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False

        # Control frame rate
        env.clock.tick(GameEnv.FPS)

    env.close()