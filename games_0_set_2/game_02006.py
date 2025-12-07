import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to jump one square. Spacebar to jump two squares "
        "in the last direction moved."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between procedurally generated platforms to reach the top. "
        "Longer jumps grant bonus points, but risk falling. Reach the top of 3 levels before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 12
    CELL_SIZE = 32
    UI_HEIGHT = SCREEN_HEIGHT - (GRID_HEIGHT * CELL_SIZE)  # 16px

    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (150, 255, 200, 40)
    COLOR_PLATFORM_BASE = (70, 130, 230)
    COLOR_GOAL = (255, 220, 50)
    COLOR_DANGER = (200, 30, 30, 100)
    COLOR_TEXT = (240, 240, 255)
    COLOR_BONUS = (255, 255, 100)

    MAX_LEVELS = 3
    TIME_PER_LEVEL = 60.0
    TIME_COST_PER_STEP = 0.25  # Each action costs time
    MAX_EPISODE_STEPS = (TIME_PER_LEVEL / TIME_COST_PER_STEP) * MAX_LEVELS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_bonus = pygame.font.Font(None, 22)

        # Initialize state variables to be populated in reset()
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.platforms = []
        self.goal_platform_idx = -1
        self.timer = 0.0
        self.level = 1
        self.level_difficulty_modifier = 1.0
        self.last_move_dir = [0, -1]  # Default up
        self.particles = []
        self.bonus_text_effects = []

        # Initialize state variables
        # self.reset() is called by the environment wrapper, not in __init__
        # self.validate_implementation() # This will be called by the test harness

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.timer = self.TIME_PER_LEVEL
        self.level_difficulty_modifier = 1.0
        self.last_move_dir = [0, -1]
        self.particles = []
        self.bonus_text_effects = []

        self._generate_platforms()
        start_platform = self.platforms[0]
        self.player_pos = [start_platform[0], start_platform[1]]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack action ---
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        # shift_held is unused per the brief's mechanics

        # --- Update game logic ---
        self.steps += 1
        self.timer -= self.TIME_COST_PER_STEP
        reward = -0.1  # Cost of existing

        old_pos = list(self.player_pos)
        jump_vector = [0, 0]
        is_risky_jump = False

        move_map = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}
        if space_held:
            # Risky jump: 2 units in last direction
            jump_vector = [d * 2 for d in self.last_move_dir]
            is_risky_jump = True
        elif movement in move_map:
            # Normal jump: 1 unit
            jump_vector = move_map[movement]
            self.last_move_dir = list(jump_vector)

        # Only process if a jump was made
        if not np.array_equal(jump_vector, [0, 0]):
            target_pos = [self.player_pos[0] + jump_vector[0], self.player_pos[1] + jump_vector[1]]

            landed_on_platform = False
            landed_on_goal = False

            for i, p in enumerate(self.platforms):
                if p[0] == target_pos[0] and p[1] == target_pos[1]:
                    landed_on_platform = True
                    self.player_pos = target_pos
                    if i == self.goal_platform_idx:
                        landed_on_goal = True
                    break

            # --- Calculate Reward and Update State ---
            if landed_on_platform:
                self._spawn_particles(old_pos, self.COLOR_PLAYER, 10, jump_vector)

                if is_risky_jump:
                    reward += 5.0
                    self.score += 50
                    self._add_bonus_text("+50", old_pos, self.COLOR_BONUS)
                else:
                    reward += -2.0
                    self.score -= 10
                    self._add_bonus_text("-10", old_pos, (255, 100, 100))

                if landed_on_goal:
                    self.level += 1
                    self.score += 100
                    reward += 10.0
                    self._add_bonus_text("LEVEL CLEAR! +100", [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2], self.COLOR_GOAL)

                    if self.level > self.MAX_LEVELS:
                        # Game Won
                        self.game_over = True
                        reward += 100.0  # Final win bonus
                        self.score += 1000
                    else:
                        # Next Level
                        self.timer = self.TIME_PER_LEVEL
                        self.level_difficulty_modifier += 0.05
                        self._generate_platforms()
                        start_platform = self.platforms[0]
                        self.player_pos = [start_platform[0], start_platform[1]]
            else:
                # Fell off
                self.game_over = True
                reward += -50.0
                self.score -= 500
                self._spawn_particles(old_pos, self.COLOR_DANGER, 30, [0, 1])

        # --- Check Termination Conditions ---
        if self.timer <= 0:
            self.game_over = True
            reward += -50.0  # Penalty for timeout

        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True

        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _generate_platforms(self):
        self.platforms.clear()

        # Start platform at bottom-center
        start_x = self.GRID_WIDTH // 2
        start_y = self.GRID_HEIGHT - 2
        current_platform = [start_x, start_y]
        self.platforms.append(current_platform)

        # Generate a guaranteed path to the top
        path_length = self.GRID_HEIGHT - 2
        for _ in range(path_length):
            # Move mostly upwards
            dy = -self.np_random.integers(1, 3)  # Jump 1 or 2 units up
            dx = self.np_random.integers(-2, 3)

            next_x = np.clip(current_platform[0] + dx, 1, self.GRID_WIDTH - 2)
            next_y = np.clip(current_platform[1] + dy, 0, self.GRID_HEIGHT - 1)

            # Avoid placing platforms directly on top of each other too often
            if next_x == current_platform[0] and next_y == current_platform[1]:
                next_y = np.clip(current_platform[1] - 1, 0, self.GRID_HEIGHT - 1)

            current_platform = [next_x, next_y]
            if current_platform not in self.platforms:
                self.platforms.append(current_platform)

        # Ensure the last platform is at the top
        self.platforms[-1][1] = 0
        self.goal_platform_idx = len(self.platforms) - 1

        # Add some random distractor platforms
        num_distractors = int(15 * self.level_difficulty_modifier)
        for _ in range(num_distractors):
            rand_x = self.np_random.integers(0, self.GRID_WIDTH)
            rand_y = self.np_random.integers(0, self.GRID_HEIGHT - 1)
            if [rand_x, rand_y] not in self.platforms:
                self.platforms.append([rand_x, rand_y])

    def _get_observation(self):
        # Update animations
        self._update_effects()

        # Render all game elements
        self._render_background()
        self._render_platforms()
        self._render_effects()
        self._render_player()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "timer": self.timer,
        }

    # --- Rendering Helpers ---

    def _grid_to_pixel(self, grid_pos):
        px = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2 + self.UI_HEIGHT
        return int(px), int(py)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.UI_HEIGHT), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE + self.UI_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

        # Draw danger zone
        danger_rect = pygame.Rect(0, self.SCREEN_HEIGHT - self.CELL_SIZE / 2, self.SCREEN_WIDTH, self.CELL_SIZE / 2)
        s = pygame.Surface((danger_rect.width, danger_rect.height), pygame.SRCALPHA)
        s.fill(self.COLOR_DANGER)
        self.screen.blit(s, (danger_rect.x, danger_rect.y))

    def _render_platforms(self):
        for i, p in enumerate(self.platforms):
            px, py = self._grid_to_pixel(p)
            rect = pygame.Rect(px - self.CELL_SIZE // 2, py - self.CELL_SIZE // 2, self.CELL_SIZE, self.CELL_SIZE)

            if i == self.goal_platform_idx:
                color = self.COLOR_GOAL
                inner_color = tuple(min(255, c + 30) for c in color)
            else:
                # Color varies with height
                height_factor = 1.0 - (p[1] / self.GRID_HEIGHT)
                color = tuple(int(c * (0.6 + 0.4 * height_factor)) for c in self.COLOR_PLATFORM_BASE)
                inner_color = tuple(min(255, c + 30) for c in color)

            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, inner_color, rect.inflate(-6, -6), border_radius=4)

    def _render_player(self):
        px, py = self._grid_to_pixel(self.player_pos)

        # Glow effect
        glow_radius = int(self.CELL_SIZE * 0.8)
        pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, glow_radius, self.COLOR_PLAYER_GLOW)

        # Player square
        rect = pygame.Rect(px - self.CELL_SIZE // 2 + 4, py - self.CELL_SIZE // 2 + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=3)
        pygame.draw.rect(self.screen, (200, 255, 220), rect.inflate(-4, -4), border_radius=3)

    def _render_effects(self):
        # Draw particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1

            alpha = max(0, int(255 * (p['life'] / p['max_life'])))
            # Take only RGB from base color and add new alpha to form a valid RGBA tuple
            color = p['color'][:3] + (alpha,)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                # Use gfxdraw for alpha blending and cast float positions to int
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

        # Draw bonus text
        for effect in self.bonus_text_effects:
            effect['pos'][1] -= 0.5  # Move up
            effect['life'] -= 1

            alpha = max(0, int(255 * (effect['life'] / effect['max_life'])))
            text_surf = self.font_bonus.render(effect['text'], True, effect['color'])
            text_surf.set_alpha(alpha)
            # Ensure center position is an integer tuple
            text_rect = text_surf.get_rect(center=(int(effect['pos'][0]), int(effect['pos'][1])))
            self.screen.blit(text_surf, text_rect)

    def _update_effects(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        self.bonus_text_effects = [e for e in self.bonus_text_effects if e['life'] > 0]

    def _spawn_particles(self, grid_pos, color, count, jump_vector):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed - jump_vector[0] * 0.5,
                   math.sin(angle) * speed - jump_vector[1] * 0.5]
            self.particles.append({
                'pos': [px, py],
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _add_bonus_text(self, text, grid_pos, color):
        px, py = self._grid_to_pixel(grid_pos)
        self.bonus_text_effects.append({
            'text': text,
            'pos': [px, py - 10],
            'color': color,
            'life': 60,
            'max_life': 60
        })

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 8))

        # Timer
        timer_text = self.font_main.render(f"TIME: {max(0, self.timer):.1f}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(right=self.SCREEN_WIDTH - 10, top=8)
        self.screen.blit(timer_text, timer_rect)

        # Level
        level_text = self.font_main.render(f"LEVEL: {self.level}/{self.MAX_LEVELS}", True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(centerx=self.SCREEN_WIDTH / 2, top=8)
        self.screen.blit(level_text, level_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_text_str = "GAME WON!" if self.level > self.MAX_LEVELS else "GAME OVER"
            end_text = pygame.font.Font(None, 72).render(end_text_str, True,
                                                          self.COLOR_GOAL if self.level > self.MAX_LEVELS else self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(end_text, end_rect)

            score_text = pygame.font.Font(None, 48).render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 40))
            self.screen.blit(score_text, score_rect)


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires pygame to be installed and will open a window.
    # The environment itself runs headlessly.
    
    # Unset the dummy video driver to allow window creation
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    done = False

    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Hopper")
    clock = pygame.time.Clock()

    # Game loop
    running = True
    while running:
        # --- Action mapping for human input ---
        movement = 0  # No-op
        space_held = 0
        shift_held = 0
        
        # Check for key presses once per frame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset on 'r'
                    obs, info = env.reset()
                    done = False
                if event.key == pygame.K_q:  # Quit on 'q'
                    running = False
                
                # Map keys to actions
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: movement = 1
                elif keys[pygame.K_DOWN]: movement = 2
                elif keys[pygame.K_LEFT]: movement = 3
                elif keys[pygame.K_RIGHT]: movement = 4
                
                if keys[pygame.K_SPACE]: space_held = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

                action = [movement, space_held, shift_held]
                
                # --- Step the environment ---
                if not done and any(k != 0 for k in action):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

        # --- Render the game state ---
        # The observation is the rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(30)  # Limit FPS for human play

    pygame.quit()