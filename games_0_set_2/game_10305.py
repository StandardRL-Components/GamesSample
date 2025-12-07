import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:10:47.939851
# Source Brief: brief_00305.md
# Brief Index: 305
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a cell.
    The goal is to survive for 60 seconds by absorbing smaller cells to grow,
    while avoiding being consumed by larger predator cells.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Control a cell and absorb smaller cells to grow while avoiding larger ones. Survive for as long as possible in a dangerous cellular world."
    user_guide = "Use the arrow keys (↑↓←→) to move your cell. Absorb smaller cells to grow and evade larger ones to survive."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 60  # For consistent game speed and physics
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = TIME_LIMIT_SECONDS * TARGET_FPS

    # --- Colors ---
    COLOR_BG = (10, 20, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PREY = (0, 192, 255)
    COLOR_PREDATOR = (255, 64, 64)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_SHADOW = (0, 0, 0)
    COLOR_PARTICLE = (40, 50, 60)

    # --- Player ---
    PLAYER_INITIAL_SIZE = 15.0
    PLAYER_BASE_SPEED = 250.0  # pixels per second

    # --- Cells ---
    INITIAL_CELL_COUNT = 15
    MAX_CELL_COUNT = 30
    CELL_SPAWN_INTERVAL = 1.0  # seconds
    DIFFICULTY_INTERVAL = 5.0  # seconds
    DIFFICULTY_SIZE_INCREASE = 0.5
    INITIAL_AVG_SPAWN_SIZE = 8.0
    
    # --- Background Particles ---
    PARTICLE_COUNT = 150
    PARTICLE_MAX_SPEED = 10 # pixels per second

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0
        self.player_pos = None
        self.player_size = 0.0
        self.cells = []
        self.particles = []
        self.avg_spawn_size = 0.0
        self.spawn_timer = 0.0
        self.difficulty_timer = 0.0
        
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() # This is for dev, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Use a numpy random number generator for consistency
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0

        # Player state
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_size = self.PLAYER_INITIAL_SIZE

        # Cell state
        self.cells = []
        self.avg_spawn_size = self.INITIAL_AVG_SPAWN_SIZE
        for _ in range(self.INITIAL_CELL_COUNT):
            self._spawn_cell(initial_spawn=True)

        self.spawn_timer = 0.0
        self.difficulty_timer = 0.0
        
        self._init_background_particles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        # space_held and shift_held are unused per the brief
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        reward = 0.0
        dt = 1.0 / self.TARGET_FPS
        self.time_elapsed += dt
        self.spawn_timer += dt
        self.difficulty_timer += dt

        # --- Game Logic Updates ---
        self._move_player(movement, dt)
        self._update_cells_and_particles(dt)
        
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        self._handle_spawning()
        self._update_difficulty()

        # --- Termination and Final Rewards ---
        self.steps += 1
        terminated = False
        truncated = False
        if self.game_over:
            terminated = True
            reward = -100.0  # Penalty for being consumed
        elif self.time_elapsed >= self.TIME_LIMIT_SECONDS:
            truncated = True # Use truncated for time limit
            reward = 100.0  # Bonus for survival
        
        # Add small survival reward per step if not terminated
        if not terminated and not truncated:
            reward += 0.1

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _move_player(self, movement, dt):
        direction = pygame.Vector2(0, 0)
        if movement == 1: direction.y = -1  # Up
        elif movement == 2: direction.y = 1   # Down
        elif movement == 3: direction.x = -1  # Left
        elif movement == 4: direction.x = 1   # Right

        # Speed is inversely proportional to size
        speed = self.PLAYER_BASE_SPEED / max(1.0, self.player_size / self.PLAYER_INITIAL_SIZE)

        if direction.length() > 0:
            direction.normalize_ip()
            self.player_pos += direction * speed * dt

        # Prevent player from going off-screen
        self.player_pos.x = np.clip(self.player_pos.x, self.player_size, self.SCREEN_WIDTH - self.player_size)
        self.player_pos.y = np.clip(self.player_pos.y, self.player_size, self.SCREEN_HEIGHT - self.player_size)

    def _update_cells_and_particles(self, dt):
        # Update enemy cells
        for cell in self.cells:
            cell['pos'] += cell['vel'] * dt
            # Bounce off walls
            if cell['pos'].x - cell['size'] < 0 or cell['pos'].x + cell['size'] > self.SCREEN_WIDTH:
                cell['vel'].x *= -1
            if cell['pos'].y - cell['size'] < 0 or cell['pos'].y + cell['size'] > self.SCREEN_HEIGHT:
                cell['vel'].y *= -1
        
        # Update background particles
        for p in self.particles:
            p['pos'] += p['vel'] * dt
            # Screen wrap
            if p['pos'].x < 0: p['pos'].x = self.SCREEN_WIDTH
            if p['pos'].x > self.SCREEN_WIDTH: p['pos'].x = 0
            if p['pos'].y < 0: p['pos'].y = self.SCREEN_HEIGHT
            if p['pos'].y > self.SCREEN_HEIGHT: p['pos'].y = 0

    def _handle_collisions(self):
        absorption_reward = 0.0
        cells_to_remove = []
        for cell in self.cells:
            distance = self.player_pos.distance_to(cell['pos'])
            # Use a slightly more generous collision radius for game feel
            if distance < self.player_size + cell['size'] - 2:
                # Player is larger (and has a small margin of safety)
                if self.player_size > cell['size'] * 1.01:
                    # Growth based on area, feels more natural
                    new_player_area = math.pi * self.player_size**2
                    consumed_area = math.pi * cell['size']**2
                    self.player_size = math.sqrt((new_player_area + consumed_area) / math.pi)
                    
                    self.score += 1
                    absorption_reward += 1.0
                    cells_to_remove.append(cell)
                else:
                    self.game_over = True
                    break # No need to check other cells
        
        if not self.game_over:
            self.cells = [c for c in self.cells if c not in cells_to_remove]
        
        return absorption_reward

    def _handle_spawning(self):
        if self.spawn_timer >= self.CELL_SPAWN_INTERVAL:
            self.spawn_timer = 0.0
            if len(self.cells) < self.MAX_CELL_COUNT:
                self._spawn_cell()

    def _update_difficulty(self):
        if self.difficulty_timer >= self.DIFFICULTY_INTERVAL:
            self.difficulty_timer = 0.0
            self.avg_spawn_size += self.DIFFICULTY_SIZE_INCREASE

    def _spawn_cell(self, initial_spawn=False):
        size_variance = self.avg_spawn_size * 0.5
        size = self.np_random.uniform(self.avg_spawn_size - size_variance, self.avg_spawn_size + size_variance)
        size = max(2.0, size) # Ensure minimum size

        if initial_spawn:
            # Spawn anywhere on screen for the initial set
            pos = pygame.Vector2(self.np_random.uniform(size, self.SCREEN_WIDTH - size),
                                 self.np_random.uniform(size, self.SCREEN_HEIGHT - size))
        else:
            # Spawn off-screen and move inwards
            edge = self.np_random.integers(0, 4)
            if edge == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -size)
            elif edge == 1: # Right
                pos = pygame.Vector2(self.SCREEN_WIDTH + size, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            elif edge == 2: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + size)
            else: # Left
                pos = pygame.Vector2(-size, self.np_random.uniform(0, self.SCREEN_HEIGHT))

        # Velocity points towards the center of the screen
        center = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        vel = (center - pos).normalize() * self.np_random.uniform(20, 50)
        
        self.cells.append({'pos': pos, 'size': size, 'vel': vel})

    def _init_background_particles(self):
        self.particles = []
        for _ in range(self.PARTICLE_COUNT):
            self.particles.append({
                'pos': pygame.Vector2(self.np_random.integers(0, self.SCREEN_WIDTH + 1), self.np_random.integers(0, self.SCREEN_HEIGHT + 1)),
                'vel': pygame.Vector2(self.np_random.uniform(-self.PARTICLE_MAX_SPEED, self.PARTICLE_MAX_SPEED), 
                                     self.np_random.uniform(-self.PARTICLE_MAX_SPEED, self.PARTICLE_MAX_SPEED)),
                'radius': self.np_random.uniform(0.5, 1.5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_size": self.player_size,
            "time_elapsed": self.time_elapsed,
        }

    def _render_game(self):
        # Render background particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), self.COLOR_PARTICLE)

        # Render enemy cells
        for cell in self.cells:
            color = self.COLOR_PREY if cell['size'] < self.player_size else self.COLOR_PREDATOR
            pos_int = (int(cell['pos'].x), int(cell['pos'].y))
            radius_int = int(cell['size'])
            if radius_int > 0:
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius_int, color)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, color)

        # Render player with glow
        player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        player_radius_int = int(self.player_size)
        if player_radius_int > 0:
            self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, player_pos_int, player_radius_int, 15)

    def _render_ui(self):
        # Render Player Size (as score)
        size_text = f"Cells Absorbed: {self.score}"
        self._draw_text(size_text, self.font_ui, self.COLOR_UI_TEXT, (10, 10), self.COLOR_UI_SHADOW)

        # Render Timer
        time_left = max(0, self.TIME_LIMIT_SECONDS - self.time_elapsed)
        timer_text = f"Time: {time_left:.1f}"
        text_width = self.font_ui.size(timer_text)[0]
        self._draw_text(timer_text, self.font_ui, self.COLOR_UI_TEXT, (self.SCREEN_WIDTH - text_width - 10, 10), self.COLOR_UI_SHADOW)

    def _draw_text(self, text, font, color, pos, shadow_color):
        shadow_pos = (pos[0] + 2, pos[1] + 2)
        text_surface_shadow = font.render(text, True, shadow_color)
        self.screen.blit(text_surface_shadow, shadow_pos)
        
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _draw_glowing_circle(self, surface, color, center, radius, max_glow_radius):
        # Draws multiple transparent circles to create a glow effect
        for i in range(max_glow_radius, 0, -2):
            alpha = 40 * (1 - (i / max_glow_radius))
            glow_color = (*color, int(alpha))
            glow_radius = radius + i
            # This can error if radius is < 0, but we check for radius > 0 before calling
            try:
                # Use a temporary surface for alpha blending
                temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
                surface.blit(temp_surf, (center[0] - glow_radius, center[1] - glow_radius))
            except pygame.error:
                # Can happen if glow_radius becomes huge or negative, just skip frame
                pass
        
        # Draw the main, solid circle on top
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit() # a bit of a hack to re-init with video
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Cell Survival - Manual Control")
    screen_display = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    terminated = False
    truncated = False
    total_reward = 0.0
    
    print("Controls: Arrow keys to move. Close window to quit.")

    while not terminated and not truncated:
        # --- Human Controls ---
        movement_action = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        # The other actions are not used in this game
        action = [movement_action, 0, 0]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            reason = "eaten" if terminated else "time limit"
            print(f"Game Over! ({reason}) Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Survived for {info['time_elapsed']:.2f}s")
            break

        # --- Rendering ---
        # The observation is already a rendered frame, we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(GameEnv.TARGET_FPS)

    env.close()