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
        "Controls: Use arrow keys to slide a block. Press space to cycle which block is selected."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Slide colored blocks to fill the target zone within a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    GRID_COLS = 16
    GRID_ROWS = 10
    CELL_SIZE = 40
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_TARGET = (60, 70, 90)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)
    COLOR_WIN = (100, 255, 150)
    COLOR_LOSE = (255, 100, 100)
    
    BLOCK_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 120, 255), # Magenta
        (80, 255, 255),  # Cyan
        (255, 160, 80),  # Orange
        (160, 80, 255),  # Purple
        (200, 200, 200), # White
    ]
    
    INITIAL_MOVES = 10
    MAX_STEPS = 1000

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
        
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 30)
            self.font_big = pygame.font.SysFont(None, 60)
            
        self.blocks = []
        self.particles = []
        self.target_area = pygame.Rect(0, 0, 0, 0)
        self.selected_block_idx = 0
        self.moves_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.rng = np.random.default_rng()
        
        # self.reset() is called by the validation function
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.INITIAL_MOVES
        self.game_over = False
        self.last_space_held = False
        self.particles = []
        
        target_w, target_h = 3, 3
        target_x = self.rng.integers(1, self.GRID_COLS - target_w - 1)
        target_y = self.rng.integers(1, self.GRID_ROWS - target_h - 1)
        self.target_area = pygame.Rect(target_x, target_y, target_w, target_h)
        
        num_blocks = target_w * target_h
        self.blocks = []
        occupied_pos = set()
        
        for i in range(num_blocks):
            while True:
                pos = (self.rng.integers(0, self.GRID_COLS), self.rng.integers(0, self.GRID_ROWS))
                pos_rect = pygame.Rect(pos[0], pos[1], 1, 1)
                if pos not in occupied_pos and not self.target_area.contains(pos_rect):
                    occupied_pos.add(pos)
                    break
            
            color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
            self.blocks.append({"pos": pygame.Vector2(pos), "color": color})

        if self.blocks:
            self.selected_block_idx = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        if space_pressed and len(self.blocks) > 1:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.blocks)
        
        if movement > 0 and self.moves_remaining > 0:
            self.moves_remaining -= 1
            
            direction = self._get_direction_from_action(movement)
            block = self.blocks[self.selected_block_idx]
            start_pos = pygame.Vector2(block["pos"])
            
            # Slide logic
            current_pos = pygame.Vector2(start_pos)
            while True:
                next_pos = current_pos + direction
                if not (0 <= next_pos.x < self.GRID_COLS and 0 <= next_pos.y < self.GRID_ROWS) \
                   or self._is_occupied(next_pos, exclude_block=block):
                    break
                current_pos = next_pos
            
            end_pos = current_pos
            
            if start_pos != end_pos:
                block["pos"] = end_pos
                self._create_trail(start_pos, end_pos, block["color"])
                # sfx: block slide
                reward = self._calculate_move_reward(block)
            else:
                # sfx: bump
                reward = -0.02 # Penalty for a wasted move
        
        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        truncated = False
        
        if terminated and not self.game_over: # First frame of termination
            self.game_over = True
            if self._is_win_condition_met():
                # sfx: win jingle
                win_reward = 50
                reward += win_reward
                self.score += win_reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_direction_from_action(self, movement_action):
        if movement_action == 1: return pygame.Vector2(0, -1)  # Up
        if movement_action == 2: return pygame.Vector2(0, 1)   # Down
        if movement_action == 3: return pygame.Vector2(-1, 0)  # Left
        if movement_action == 4: return pygame.Vector2(1, 0)   # Right
        return pygame.Vector2(0, 0)

    def _is_occupied(self, pos, exclude_block=None):
        for b in self.blocks:
            if b is exclude_block:
                continue
            if b["pos"] == pos:
                return True
        return False

    def _calculate_move_reward(self, block):
        block_rect = pygame.Rect(int(block["pos"].x), int(block["pos"].y), 1, 1)
        
        if self.target_area.contains(block_rect):
            # sfx: place block in target
            return 1.0
        
        # Check for adjacency using an inflated rect
        if self.target_area.colliderect(block_rect.inflate(2, 2)):
            return 0.1
            
        return -0.02

    def _is_win_condition_met(self):
        if not self.blocks:
            return False
            
        target_cells = set()
        for x in range(self.target_area.left, self.target_area.right):
            for y in range(self.target_area.top, self.target_area.bottom):
                target_cells.add((x, y))
        
        block_positions = { (int(b["pos"].x), int(b["pos"].y)) for b in self.blocks }
        
        return target_cells.issubset(block_positions)

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if self._is_win_condition_met():
            return True
        if self.moves_remaining <= 0:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_draw_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw target area
        target_px_rect = pygame.Rect(
            self.target_area.x * self.CELL_SIZE,
            self.target_area.y * self.CELL_SIZE,
            self.target_area.width * self.CELL_SIZE,
            self.target_area.height * self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_TARGET, target_px_rect)

        # Draw blocks
        for i, block in enumerate(self.blocks):
            px, py = int(block["pos"].x * self.CELL_SIZE), int(block["pos"].y * self.CELL_SIZE)
            block_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
            
            # Inner fill
            pygame.draw.rect(self.screen, block["color"], block_rect.inflate(-4, -4))
            
            # Border
            pygame.draw.rect(self.screen, block["color"], block_rect, 2)

            if i == self.selected_block_idx and not self.game_over:
                # Selection highlight
                pygame.draw.rect(self.screen, (255, 255, 255), block_rect, 3)
                # Draw a small animated pulse effect
                pulse_alpha = 100 + 80 * math.sin(pygame.time.get_ticks() * 0.01)
                s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                s.fill((255, 255, 255, int(pulse_alpha)))
                self.screen.blit(s, (px, py), special_flags=pygame.BLEND_RGBA_ADD)


    def _create_trail(self, start_pos, end_pos, color):
        dist = start_pos.distance_to(end_pos)
        if dist == 0: return
        
        for i in np.linspace(0, 1, int(dist * 3)): # 3 particles per cell
            pos = start_pos.lerp(end_pos, i)
            px_pos = (pos.x * self.CELL_SIZE + self.CELL_SIZE / 2, 
                      pos.y * self.CELL_SIZE + self.CELL_SIZE / 2)
            
            self.particles.append({
                "pos": list(px_pos),
                "radius": self.rng.uniform(4, 8),
                "color": color,
                "life": self.rng.integers(15, 25),
                "decay": self.rng.uniform(0.85, 0.95)
            })

    def _update_and_draw_particles(self):
        for p in self.particles:
            p["life"] -= 1
            p["radius"] *= p["decay"]
            
            if p["life"] > 0 and p["radius"] > 1:
                # Using gfxdraw for anti-aliased circles
                pygame.gfxdraw.aacircle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), p["color"])
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), p["color"])
        
        self.particles = [p for p in self.particles if p["life"] > 0 and p["radius"] > 1]

    def _render_text(self, text, pos, font, color, shadow_color):
        shadow_surf = font.render(text, True, shadow_color)
        self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        moves_text = f"Moves: {self.moves_remaining}"
        score_text = f"Score: {self.score:.2f}"
        
        self._render_text(moves_text, (10, 10), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self._render_text(score_text, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            is_win = self._is_win_condition_met()
            message = "YOU WIN!" if is_win else "OUT OF MOVES"
            color = self.COLOR_WIN if is_win else self.COLOR_LOSE
            
            text_surf = self.font_big.render(message, True, color)
            pos = (self.SCREEN_WIDTH / 2 - text_surf.get_width() / 2, 
                   self.SCREEN_HEIGHT / 2 - text_surf.get_height() / 2)
            self.screen.blit(text_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "is_win": self._is_win_condition_met() if self.game_over else False
        }
        
    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment
    try:
        del os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        pass
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print(f"GAME: Block Pusher")
    print(f"  - {env.game_description}")
    print(f"  - {env.user_guide}")
    print("="*30 + "\n")

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Pusher")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Human Controls ---
        action_to_send = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action_to_send = (1, 0, 0)
                elif event.key == pygame.K_DOWN: action_to_send = (2, 0, 0)
                elif event.key == pygame.K_LEFT: action_to_send = (3, 0, 0)
                elif event.key == pygame.K_RIGHT: action_to_send = (4, 0, 0)
                elif event.key == pygame.K_SPACE: action_to_send = (0, 1, 0)
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
                    print("\n--- GAME RESET ---")
        
        if action_to_send is not None:
             obs, reward, terminated, truncated, info = env.step(action_to_send)
             print(f"Action: {action_to_send}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_remaining']}")
             if terminated:
                 print("--- GAME OVER ---")
                 print(f"Final Score: {info['score']:.2f}, Win: {info['is_win']}")
                 pygame.time.wait(2000)
                 obs, info = env.reset()
                 print("\n--- NEW GAME ---")

        # --- Rendering ---
        frame = env.render()
        # The observation is (H, W, C), but pygame screen expects (W, H)
        # and surfarray.make_surface expects transposed (W, H) array
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit to 30 FPS

    env.close()