
# Generated: 2025-08-27T20:49:44.140655
# Source Brief: brief_02590.md
# Brief Index: 2590

        
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


# --- Helper Classes for Game Logic ---

class Interactive:
    """Base class for anything the player can interact with."""
    def __init__(self, pos, size, hint=""):
        self.pos = pos  # Grid position (x, y)
        self.size = size # Visual size for rendering
        self.hint = hint
        self.is_solved = False
        self.is_highlighted = False
        self.pulse_alpha = 0

    def get_screen_pos(self, iso_converter):
        return iso_converter(self.pos[0], self.pos[1])

    def interact(self, game_state):
        """Returns a reward for the interaction."""
        return 0 # Default: no reward

    def update(self):
        """Called every frame for animations."""
        if self.is_highlighted:
            self.pulse_alpha = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 * 150 + 50
        else:
            self.pulse_alpha = 0

    def draw(self, screen, iso_converter, font):
        pass # To be implemented by subclasses

class Lever(Interactive):
    """A single lever in a puzzle."""
    def __init__(self, pos, lever_id):
        super().__init__(pos, (10, 40), hint="Pull Lever")
        self.id = lever_id
        self.is_pulled = False

    def draw(self, screen, iso_converter, font):
        x, y = self.get_screen_pos(iso_converter)
        base_rect = pygame.Rect(x - 5, y - 5, 10, 10)
        pygame.draw.rect(screen, (80, 80, 90), base_rect)
        pygame.draw.rect(screen, (50, 50, 60), base_rect, 2)

        if self.is_pulled:
            stick_end = (x + 15, y - 15)
            color = (100, 255, 100)
        else:
            stick_end = (x - 15, y - 15)
            color = (255, 100, 100)
        pygame.draw.line(screen, (50, 50, 60), (x, y), stick_end, 6)
        pygame.draw.line(screen, color, (x, y), stick_end, 4)

class LeverPuzzle(Interactive):
    """Manages a sequence of levers."""
    def __init__(self, positions, solution_sequence):
        # This is a logical-only interactive, position doesn't matter
        super().__init__((-1, -1), (0, 0))
        self.levers = [Lever(pos, i) for i, pos in enumerate(positions)]
        self.solution = solution_sequence
        self.current_step = 0

    def interact(self, game_state, lever_id):
        # This method is called from the GameEnv, not directly by player
        lever = next((l for l in self.levers if l.id == lever_id), None)
        if not lever or lever.is_pulled:
            return 0 # Can't pull a pulled lever

        if self.solution[self.current_step] == lever.id:
            lever.is_pulled = True
            self.current_step += 1
            if self.current_step == len(self.solution):
                self.is_solved = True
                # sfx: puzzle_solved_chime
                return 5.0 # Big reward for solving
            # sfx: lever_correct_click
            return 0.1 # Small reward for correct step
        else:
            # sfx: puzzle_error_buzz
            self.reset_levers()
            return -0.1 # Small penalty for mistake

    def reset_levers(self):
        self.current_step = 0
        for lever in self.levers:
            lever.is_pulled = False

    def get_interactives(self):
        return self.levers

class Door(Interactive):
    """A door that can be locked or unlocked."""
    def __init__(self, pos, leads_to, puzzle_to_solve=None):
        super().__init__(pos, (30, 60), hint="Open Door")
        self.leads_to = leads_to # room index or "win"
        self.puzzle = puzzle_to_solve
        self.is_locked = puzzle_to_solve is not None
        self.is_open = False

    def update(self):
        super().update()
        if self.is_locked and self.puzzle and self.puzzle.is_solved:
            self.is_locked = False
            self.hint = "Enter"

    def interact(self, game_state):
        if not self.is_locked and not self.is_open:
            self.is_open = True
            # sfx: door_creak_open
            if self.leads_to == "win":
                game_state.win_game()
                return 50.0
            else:
                game_state.change_room(self.leads_to)
                return 2.0
        elif self.is_locked:
            # sfx: door_locked_rattle
            return -0.05
        return 0

    def draw(self, screen, iso_converter, font):
        x, y = self.get_screen_pos(iso_converter)
        color = (139, 69, 19) if not self.is_locked else (90, 90, 100)
        if self.is_open:
            color = (20, 20, 20)

        rect = pygame.Rect(0, 0, self.size[0], self.size[1])
        rect.center = (x, y - self.size[1] // 2)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (50, 50, 60), rect, 3)

        if not self.is_open:
            knob_pos = (x + 10, y - self.size[1] // 2 + 30)
            pygame.draw.circle(screen, (255, 223, 0), knob_pos, 4)

class Room:
    """Contains all elements of a single room."""
    def __init__(self, name, puzzle, doors, floor_tiles=(10, 10)):
        self.name = name
        self.puzzle = puzzle
        self.doors = doors
        self.interactives = []
        if self.puzzle:
            self.interactives.extend(self.puzzle.get_interactives())
        self.interactives.extend(self.doors)
        self.floor_tiles = floor_tiles

# --- Main Game Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to select an object. Space to interact."
    )
    game_description = (
        "Escape a procedurally generated haunted house by solving puzzles within a time limit. A dark, atmospheric isometric puzzler."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.screen_size = (640, 400)
        self.screen = pygame.Surface(self.screen_size)
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Colors and Style ---
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_WALL = (40, 45, 60)
        self.COLOR_FLOOR = (60, 65, 80)
        self.COLOR_GRID = (50, 55, 70)
        self.COLOR_HIGHLIGHT = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 30)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_time = 1000
        self.timer = self.max_time
        self.house = []
        self.current_room_index = 0
        self.selected_interactive_index = 0
        self.particles = []

        self.reset()
        self.validate_implementation()


    def _generate_house(self):
        """Creates the layout and puzzles for the house."""
        # Room 1: Lever Puzzle
        lever_puzzle = LeverPuzzle(
            positions=[(2, 8), (4, 8), (6, 8)],
            solution_sequence=[0, 2, 1]
        )
        door_to_room2 = Door(pos=(9, 4), leads_to=1, puzzle_to_solve=lever_puzzle)
        room1 = Room("The Antechamber", lever_puzzle, [door_to_room2])

        # Room 2: "Puzzle" is just finding the exit door
        exit_door = Door(pos=(5, 0), leads_to="win")
        door_to_room1 = Door(pos=(5, 9), leads_to=0)
        room2 = Room("The Hallway", None, [exit_door, door_to_room1], floor_tiles=(10,10))

        self.house = [room1, room2]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.max_time

        self._generate_house()
        self.current_room_index = 0
        self.selected_interactive_index = 0
        self._create_particles(100)

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_press, _ = action
        reward = 0
        terminated = False
        
        # 1. Handle Input
        if movement != 0: # 0 is no-op
            self._move_cursor(movement)
        
        if space_press == 1:
            # sfx: ui_click
            reward += self._interact_with_selection()

        # 2. Update Game State
        self.timer -= 1
        for p in self.house:
            if p.puzzle: p.puzzle.is_solved = p.puzzle.is_solved
            for i in p.interactives:
                i.update()
        self._update_particles()

        # 3. Check for Termination
        if self.timer <= 0 or self.game_over:
            terminated = True
            if self.timer <= 0 and not self.game_over:
                # sfx: game_over_timer
                pass # No specific penalty, loss of win reward is enough

        # 4. Update score
        self.score += reward
        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _interact_with_selection(self):
        room = self.house[self.current_room_index]
        if not room.interactives: return 0
        
        selected_obj = room.interactives[self.selected_interactive_index]

        if isinstance(selected_obj, Lever):
            return room.puzzle.interact(self, selected_obj.id)
        elif isinstance(selected_obj, Door):
            return selected_obj.interact(self)
        return 0

    def _move_cursor(self, direction):
        # 1:up, 2:down, 3:left, 4:right
        room = self.house[self.current_room_index]
        if len(room.interactives) <= 1:
            return

        current_obj = room.interactives[self.selected_interactive_index]
        cx, cy = current_obj.pos

        best_target_idx = -1
        min_dist = float('inf')

        # Define direction vectors in grid space
        dir_vectors = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        target_vec = dir_vectors[direction]

        for i, other_obj in enumerate(room.interactives):
            if i == self.selected_interactive_index:
                continue

            ox, oy = other_obj.pos
            dist_vec = (ox - cx, oy - cy)
            dist = math.hypot(dist_vec[0], dist_vec[1])

            if dist == 0: continue

            # Project distance vector onto target direction vector
            dot_product = dist_vec[0] * target_vec[0] + dist_vec[1] * target_vec[1]

            # We want targets in the right general direction
            if dot_product > 0:
                # Penalize targets that are not aligned with the direction
                # Angle is acos(dot_product / dist), so small angle is good.
                # We use dot_product/dist which is cos(angle). 1 is perfect alignment.
                alignment = dot_product / dist
                score = dist / (alignment**2 + 0.1) # Prioritize alignment

                if score < min_dist:
                    min_dist = score
                    best_target_idx = i

        if best_target_idx != -1:
            self.selected_interactive_index = best_target_idx

    def change_room(self, room_index):
        self.current_room_index = room_index
        self.selected_interactive_index = 0

    def win_game(self):
        self.game_over = True
        # sfx: victory_fanfare

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Rendering Methods ---

    def _iso_converter(self, room):
        tile_w, tile_h = 32, 16
        room_w, room_h = room.floor_tiles
        origin_x = self.screen_size[0] / 2
        origin_y = self.screen_size[1] / 2 + 50
        
        def convert(x, y):
            screen_x = origin_x + (x - y) * tile_w / 2
            screen_y = origin_y + (x + y) * tile_h / 2
            return int(screen_x), int(screen_y)
        return convert

    def _render_text(self, text, pos, font, color, shadow_color):
        text_surf = font.render(text, True, shadow_color)
        self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_game(self):
        room = self.house[self.current_room_index]
        iso_conv = self._iso_converter(room)
        
        # Render floor and walls
        self._render_room_structure(room, iso_conv)
        self._render_particles() # Render particles behind interactives

        # Render interactives
        for i, obj in enumerate(room.interactives):
            obj.is_highlighted = (i == self.selected_interactive_index)
            obj.draw(self.screen, iso_conv, self.font_small)
            if obj.is_highlighted:
                self._render_highlight(obj, iso_conv)

    def _render_room_structure(self, room, iso_conv):
        w, h = room.floor_tiles
        # Floor points
        p = [iso_conv(0,0), iso_conv(w,0), iso_conv(w,h), iso_conv(0,h)]
        pygame.gfxdraw.filled_polygon(self.screen, p, self.COLOR_FLOOR)

        # Grid lines
        for i in range(w + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, iso_conv(i, 0), iso_conv(i, h))
        for i in range(h + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, iso_conv(0, i), iso_conv(w, i))

        # Walls
        wall_height = 150
        p_back_left = [iso_conv(0,0), iso_conv(w,0), (iso_conv(w,0)[0], iso_conv(w,0)[1]-wall_height), (iso_conv(0,0)[0], iso_conv(0,0)[1]-wall_height)]
        p_back_right = [iso_conv(w,0), iso_conv(w,h), (iso_conv(w,h)[0], iso_conv(w,h)[1]-wall_height), (iso_conv(w,0)[0], iso_conv(w,0)[1]-wall_height)]
        pygame.gfxdraw.filled_polygon(self.screen, p_back_right, self.COLOR_WALL)
        pygame.gfxdraw.filled_polygon(self.screen, p_back_left, tuple(max(0, c-10) for c in self.COLOR_WALL))
        
        pygame.gfxdraw.aapolygon(self.screen, p, self.COLOR_GRID)
        pygame.gfxdraw.aapolygon(self.screen, p_back_right, self.COLOR_GRID)
        pygame.gfxdraw.aapolygon(self.screen, p_back_left, self.COLOR_GRID)

    def _render_highlight(self, obj, iso_conv):
        x, y = obj.get_screen_pos(iso_conv)
        
        # Pulsing glow
        radius = int(max(obj.size) * 0.7)
        if obj.pulse_alpha > 0:
            glow_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_HIGHLIGHT, int(obj.pulse_alpha/2)), (radius, radius), radius)
            pygame.draw.circle(glow_surf, (*self.COLOR_HIGHLIGHT, int(obj.pulse_alpha)), (radius, radius), int(radius*0.7))
            self.screen.blit(glow_surf, (x - radius, y - radius))

        # Hint text
        if obj.hint:
            self._render_text(obj.hint, (x + 20, y - 40), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

    def _render_ui(self):
        # Timer
        timer_text = f"TIME: {self.timer}"
        self._render_text(timer_text, (self.screen_size[0] - 150, 10), self.font_large, (255, 80, 80), self.COLOR_TEXT_SHADOW)

        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._render_text(score_text, (20, 10), self.font_large, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Room Name
        room_name = self.house[self.current_room_index].name
        self._render_text(room_name, (20, self.screen_size[1] - 30), self.font_large, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        if self.game_over:
            if self.score > 0:
                msg = "ESCAPED!"
                color = (100, 255, 100)
            else:
                msg = "TIME'S UP!"
                color = (255, 100, 100)
            self._render_text(msg, (250, 180), self.font_large, color, self.COLOR_TEXT_SHADOW)

    def _create_particles(self, num):
        self.particles = []
        for _ in range(num):
            self.particles.append([
                random.uniform(0, self.screen_size[0]),
                random.uniform(0, self.screen_size[1]),
                random.uniform(0.5, 2), # speed
                random.uniform(1, 3) # size
            ])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[2] * 0.1
            p[1] += p[2] * 0.05
            if p[0] > self.screen_size[0]: p[0] = 0
            if p[1] > self.screen_size[1]: p[1] = 0

    def _render_particles(self):
        # Flickering light effect
        light_alpha = 5 + random.randint(0, 10)
        pygame.gfxdraw.filled_circle(self.screen, 320, 150, 200, (80, 80, 50, light_alpha))

        # Dust motes
        for x, y, speed, size in self.particles:
            alpha = int(speed * 20)
            pygame.draw.circle(self.screen, (200, 200, 220, alpha), (int(x), int(y)), int(size))

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode(env.screen_size)
    pygame.display.set_caption("Haunted House Escape")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # no-op, released, released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        # Only step if an action was taken
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Press 'R' to restart or 'Q' to quit.")

        # Render the observation from the environment
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Human-playable frame rate

    pygame.quit()