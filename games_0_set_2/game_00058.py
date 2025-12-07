
# Generated: 2025-08-27T12:28:35.204535
# Source Brief: brief_00058.md
# Brief Index: 58

        
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

    # A short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to move on the grid. Collect all notes before you run out of moves."
    )

    # A short, user-facing description of the game
    game_description = (
        "A beat-driven puzzle game. Navigate a vibrant grid to collect all musical notes within a limited number of moves."
    )

    # Frames only advance when an action is received
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    
    TOTAL_NOTES = 20
    MOVE_LIMIT = 30
    
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150)
    COLOR_NOTE = (255, 255, 0)
    COLOR_NOTE_GLOW = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_WARN = (255, 80, 80)
    
    class Particle:
        def __init__(self, pos, vel, radius, color, lifespan):
            self.pos = list(pos)
            self.vel = list(vel)
            self.radius = radius
            self.color = color
            self.lifespan = lifespan
            self.max_lifespan = lifespan

        def update(self):
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]
            self.radius *= 0.95
            self.lifespan -= 1

        def draw(self, surface):
            if self.lifespan > 0 and self.radius > 1:
                alpha = int(255 * (self.lifespan / self.max_lifespan))
                try:
                    pygame.gfxdraw.filled_circle(
                        surface, int(self.pos[0]), int(self.pos[1]),
                        int(self.radius), (*self.color, alpha)
                    )
                    pygame.gfxdraw.aacircle(
                        surface, int(self.pos[0]), int(self.pos[1]),
                        int(self.radius), (*self.color, alpha)
                    )
                except OverflowError: # Catches potential errors with large radius/alpha values
                    pass


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.player_pos = [0, 0]
        self.notes = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 0
        self.notes_collected = 0
        self.beat_pulse = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MOVE_LIMIT
        self.notes_collected = 0
        self.particles = []
        self.beat_pulse = 0
        
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        possible_positions = [
            (x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)
        ]
        possible_positions.remove(tuple(self.player_pos))
        
        note_indices = self.np_random.choice(
            len(possible_positions), self.TOTAL_NOTES, replace=False
        )
        self.notes = [list(possible_positions[i]) for i in note_indices]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        self.steps += 1
        self.beat_pulse += 1

        dist_before = self._find_nearest_note_dist(self.player_pos)
        
        moved = False
        if movement > 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right

            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

            if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
                self.player_pos = new_pos
                self.moves_remaining -= 1
                moved = True

        # Note collection
        if self.player_pos in self.notes:
            self.notes.remove(self.player_pos)
            self.notes_collected += 1
            reward += 10
            self.score += 10
            # // Sound effect: Note collect
            self._spawn_particles(self._grid_to_pixels(self.player_pos), self.COLOR_NOTE)

        # Proximity reward
        if moved:
            dist_after = self._find_nearest_note_dist(self.player_pos)
            if dist_after < dist_before:
                reward += 1.0
            else:
                reward -= 0.1
        
        # Update particles
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

        # Check termination conditions
        terminated = False
        if self.notes_collected == self.TOTAL_NOTES:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
            # // Sound effect: Win
        elif self.moves_remaining <= 0:
            reward -= 50
            self.score -= 50
            terminated = True
            self.game_over = True
            # // Sound effect: Lose

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _find_nearest_note_dist(self, pos):
        if not self.notes:
            return 0
        min_dist = float('inf')
        for note_pos in self.notes:
            dist = abs(pos[0] - note_pos[0]) + abs(pos[1] - note_pos[1])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(3, 8)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append(self.Particle(pos, vel, radius, color, lifespan))

    def _grid_to_pixels(self, grid_pos):
        x = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid()
        self._render_notes()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        pulse = (math.sin(self.beat_pulse * 0.2) + 1) / 2  # 0 to 1 sine wave
        
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            start_pos = (px, 0)
            end_pos = (px, self.SCREEN_HEIGHT)
            color = [int(c + pulse * 20) for c in self.COLOR_GRID]
            pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            start_pos = (0, py)
            end_pos = (self.SCREEN_WIDTH, py)
            color = [int(c + pulse * 20) for c in self.COLOR_GRID]
            pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

    def _render_notes(self):
        radius = int(self.CELL_SIZE * 0.3)
        glow_radius = int(radius * 1.8)
        for note_pos in self.notes:
            px, py = self._grid_to_pixels(note_pos)
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, (*self.COLOR_NOTE_GLOW, 20))
            # Core
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_NOTE)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_NOTE)

    def _render_player(self):
        px, py = self._grid_to_pixels(self.player_pos)
        radius = int(self.CELL_SIZE * 0.35)
        glow_radius = int(radius * 2.5)

        # Pulsating glow
        pulse = (math.sin(self.beat_pulse * 0.4) + 1) / 2
        current_glow_radius = int(glow_radius * (1 + pulse * 0.2))
        alpha = int(30 + pulse * 20)
        
        pygame.gfxdraw.filled_circle(self.screen, px, py, current_glow_radius, (*self.COLOR_PLAYER_GLOW, alpha))
        
        # Player core
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_PLAYER)

    def _render_particles(self):
        # Create a temporary surface for additive blending
        particle_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for p in self.particles:
            p.draw(particle_surface)
        self.screen.blit(particle_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Notes collected
        notes_text = f"NOTES: {self.notes_collected} / {self.TOTAL_NOTES}"
        text_surface = self.font_main.render(notes_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (20, 10))

        # Moves remaining
        moves_color = self.COLOR_TEXT if self.moves_remaining > 5 else self.COLOR_TEXT_WARN
        moves_text = f"MOVES: {self.moves_remaining}"
        text_surface = self.font_main.render(moves_text, True, moves_color)
        text_rect = text_surface.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(text_surface, text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            message = "LEVEL COMPLETE!" if self.notes_collected == self.TOTAL_NOTES else "OUT OF MOVES"
            msg_surface = self.font_main.render(message, True, self.COLOR_TEXT)
            msg_rect = msg_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surface, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "notes_collected": self.notes_collected,
            "moves_remaining": self.moves_remaining,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    
    print(env.user_guide)

    while not terminated:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                # We only process one keydown event to trigger a step
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

                if terminated:
                    print("Game Over!")

        # Draw the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human play

    env.close()