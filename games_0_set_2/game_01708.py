import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Sokoban-style puzzle game where the player pushes boxes onto target locations.

    The game is turn-based, with each action consuming one move. The goal is to
    solve each level by placing all boxes on their targets within a move limit.
    Visual feedback is provided for key actions, and the game features a retro
    pixel art style.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Push boxes onto the green 'X' targets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push pixelated boxes onto target locations within a limited number of "
        "moves across increasingly complex grid-based levels."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    CELL_SIZE = 40
    
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_WALL = (80, 80, 100)
    COLOR_WALL_TOP = (120, 120, 140)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_BOX = (50, 100, 255)
    COLOR_TARGET = (50, 200, 50)
    COLOR_BOX_ON_TARGET = (100, 220, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_POPUP_GOOD = (100, 255, 100)
    COLOR_POPUP_BAD = (255, 100, 100)
    
    LEVEL_MAPS = [
        # Level 1
        """
        WWWWWWWWWWWWWWWW
        W              W
        W P B      T   W
        W              W
        W   B      T   W
        W              W
        W     B  T     W
        W              W
        W              W
        WWWWWWWWWWWWWWWW
        """,
        # Level 2
        """
        WWWWWWWWWWWWWWWW
        W    T T       W
        W    B B     P W
        W    T T       W
        W  WWWWWWWWWW  W
        W  W         W W
        W  W B B     W W
        W  W T T     W W
        W  WWWWWWWWWW  W
        WWWWWWWWWWWWWWWW
        """,
        # Level 3
        """
        WWWWWWWWWWWWWWWW
        W T  W  B  W  TW
        W B  W     W  BW
        W    B     B   W
        W T WWW B WWW TWW
        W B  P  B  B  BW
        W    W     W   W
        W T  W  B  W  TW
        W B  W     W  BW
        WWWWWWWWWWWWWWWW
        """,
        # Level 4
        """
        WWWWWWWWWWWWWWWW
        WP W  T   T  W W
        W  W B W B W B W
        W    W W W W   W
        W T  B W B W T W
        W B  W B W B  BW
        W    W W W W   W
        W T  B W B W T W
        W  WWWWWWWWWW  W
        WWWWWWWWWWWWWWWW
        """
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.player_pos = [0, 0]
        self.box_positions = []
        self.target_positions = set()
        self.wall_positions = set()
        self.moves_left = 0
        self.current_level = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.level_complete_frames = 0
        
        self.particles = []
        self.popup_texts = []
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and "level" in options:
            self.current_level = options["level"] % len(self.LEVEL_MAPS)
        elif self.game_over: # If game over, reset to level 1
            self.current_level = 0

        self.score = 0
        self.steps = 0
        self.game_over = False
        self.level_complete_frames = 0
        self.particles.clear()
        self.popup_texts.clear()
        
        self._load_level(self.current_level)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = 0
        terminated = False
        truncated = False

        if self.level_complete_frames > 0:
            self.level_complete_frames -=1
            if self.level_complete_frames == 0:
                self.current_level = (self.current_level + 1) % len(self.LEVEL_MAPS)
                self._load_level(self.current_level)
            return self._get_observation(), 0, False, False, self._get_info()

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        if movement != 0: # 0 is no-op
            self.moves_left -= 1
            reward -= 0.1 # Cost of moving

            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            next_player_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

            # Check for wall collision
            if tuple(next_player_pos) in self.wall_positions:
                pass # Can't move into a wall
            
            # Check for box push
            elif next_player_pos in self.box_positions:
                box_idx = self.box_positions.index(next_player_pos)
                next_box_pos = [next_player_pos[0] + dx, next_player_pos[1] + dy]
                
                # Check if box can be pushed
                if tuple(next_box_pos) not in self.wall_positions and next_box_pos not in self.box_positions:
                    # Calculate rewards before moving
                    box_was_on_target = tuple(self.box_positions[box_idx]) in self.target_positions
                    box_is_now_on_target = tuple(next_box_pos) in self.target_positions

                    # Move player and box
                    self.player_pos = next_player_pos
                    self.box_positions[box_idx] = next_box_pos

                    # --- Reward Calculation ---
                    if box_is_now_on_target and not box_was_on_target:
                        reward += 5
                        self._create_particles(next_box_pos, self.COLOR_BOX_ON_TARGET, 20)
                        self._create_popup("+5", next_box_pos, self.COLOR_POPUP_GOOD)
                    elif not box_is_now_on_target and box_was_on_target:
                        reward -= 2
                        self._create_popup("-2", next_box_pos, self.COLOR_POPUP_BAD)
                    
                    # Risky/Safe move reward
                    nx, ny = next_box_pos
                    if nx <= 1 or nx >= self.GRID_COLS - 2 or ny <= 1 or ny >= self.GRID_ROWS - 2:
                        reward += 1 # Risky move bonus
                        self._create_popup("+1 Risky", next_box_pos, self.COLOR_POPUP_GOOD)
                    else:
                        reward -= 0.2 # Safe move penalty
            
            # Empty space move
            else:
                self.player_pos = next_player_pos
        
        self.score += reward

        # Check for win condition
        if self._check_win():
            reward += 100
            self.score += 100
            terminated = True
            self.level_complete_frames = 60 # Pause for 2 seconds (at 30fps)
        
        # Check for lose condition
        elif self.moves_left <= 0:
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = True
        
        if self.steps >= 1000:
            truncated = True
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _load_level(self, level_idx):
        self.player_pos = [0, 0]
        self.box_positions.clear()
        self.target_positions.clear()
        self.wall_positions.clear()
        
        level_data = self.LEVEL_MAPS[level_idx].strip().split('\n')
        for r, row_str in enumerate(level_data):
            for c, char in enumerate(row_str):
                pos = (c, r)
                if char == 'P':
                    self.player_pos = [c, r]
                elif char == 'B':
                    self.box_positions.append([c, r])
                elif char == 'T':
                    self.target_positions.add(pos)
                elif char == 'W':
                    self.wall_positions.add(pos)
                elif char == 'D': # Box on a target
                    self.box_positions.append([c, r])
                    self.target_positions.add(pos)
        
        self.moves_left = 30 + len(self.box_positions) * 5
        self.initial_moves = self.moves_left

    def _check_win(self):
        return all(tuple(box) in self.target_positions for box in self.box_positions)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._update_and_draw_effects()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level + 1,
            "moves_left": self.moves_left,
        }
        
    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw targets
        for tx, ty in self.target_positions:
            x, y = tx * self.CELL_SIZE, ty * self.CELL_SIZE
            pygame.gfxdraw.filled_circle(self.screen, x + self.CELL_SIZE // 2, y + self.CELL_SIZE // 2, self.CELL_SIZE // 3, (*self.COLOR_TARGET, 50))
            pygame.draw.line(self.screen, self.COLOR_TARGET, (x + 10, y + 10), (x + self.CELL_SIZE - 10, y + self.CELL_SIZE - 10), 3)
            pygame.draw.line(self.screen, self.COLOR_TARGET, (x + 10, y + self.CELL_SIZE - 10), (x + self.CELL_SIZE - 10, y + 10), 3)

        # Draw walls
        for wx, wy in self.wall_positions:
            rect = pygame.Rect(wx * self.CELL_SIZE, wy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
            pygame.draw.rect(self.screen, self.COLOR_WALL_TOP, (rect.left, rect.top, self.CELL_SIZE, 4))
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw boxes
        for bx, by in self.box_positions:
            rect = pygame.Rect(bx * self.CELL_SIZE + 4, by * self.CELL_SIZE + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
            is_on_target = tuple([bx, by]) in self.target_positions
            color = self.COLOR_BOX_ON_TARGET if is_on_target else self.COLOR_BOX
            
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            if is_on_target:
                pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, self.CELL_SIZE // 4, (*self.COLOR_BG, 100))
                pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, self.CELL_SIZE // 4, self.COLOR_BG)

        # Draw player
        px, py = self.player_pos
        rect = pygame.Rect(px * self.CELL_SIZE + 6, py * self.CELL_SIZE + 6, self.CELL_SIZE - 12, self.CELL_SIZE - 12)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=6)
        pygame.draw.rect(self.screen, (255, 150, 150), (rect.left, rect.top, rect.width, 4), border_radius=6)

    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill((20, 20, 30, 200))
        self.screen.blit(ui_panel, (0, 0))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 40), (self.WIDTH, 40))

        # Render texts
        level_text = self.font_small.render(f"Level: {self.current_level + 1}", True, self.COLOR_TEXT)
        moves_text = self.font_small.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        
        self.screen.blit(level_text, (10, 12))
        self.screen.blit(moves_text, (self.WIDTH // 2 - moves_text.get_width() // 2, 12))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 12))

        # Render end-game messages
        if self.level_complete_frames > 0:
            self._render_centered_text("LEVEL CLEAR!", self.font_large, self.COLOR_POPUP_GOOD)
        elif self.game_over:
            self._render_centered_text("GAME OVER", self.font_large, self.COLOR_POPUP_BAD)

    def _render_centered_text(self, text, font, color):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        
        # Add a dark background for readability
        bg_rect = text_rect.inflate(20, 20)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill((*self.COLOR_BG, 220))
        self.screen.blit(bg_surf, bg_rect)
        
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, grid_pos, color, count):
        px, py = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2, grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": [px, py],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(20, 40),
                "color": color,
                "size": random.uniform(2, 5)
            })

    def _create_popup(self, text, grid_pos, color):
        px, py = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2, grid_pos[1] * self.CELL_SIZE
        self.popup_texts.append({
            "text": text,
            "pos": [px, py],
            "life": 50,
            "color": color
        })
        
    def _update_and_draw_effects(self):
        # Update and draw particles
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p["life"] / 40))))
                pygame.draw.circle(self.screen, (*p["color"], alpha), p["pos"], p["size"])
        
        # Update and draw popups
        for pt in self.popup_texts[:]:
            pt["pos"][1] -= 0.5 # move up
            pt["life"] -= 1
            if pt["life"] <= 0:
                self.popup_texts.remove(pt)
            else:
                alpha = max(0, min(255, int(255 * (pt["life"] / 50))))
                text_surf = self.font_small.render(pt["text"], True, pt["color"])
                text_surf.set_alpha(alpha)
                text_rect = text_surf.get_rect(center=pt["pos"])
                self.screen.blit(text_surf, text_rect)

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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    # To do so, you might need to unset the dummy video driver
    # and create a real display.
    # For example:
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    
    # For automated testing, the headless mode is fine.
    try:
        env = GameEnv()
        env.validate_implementation()
        env.close()
    except Exception as e:
        print(f"Error during validation: {e}")


    # Manual Play block
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Sokoban Grid")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    action = [0,0,0]
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        if any(a != 0 for a in action):
             obs, reward, terminated, truncated, info = env.step(action)
             print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
             if terminated:
                 print("--- Episode Finished ---")
                 # In a real scenario, you might auto-reset or wait for a key press
                 # For this manual test, we'll let the "GAME OVER" or "LEVEL CLEAR" message display
                 # and the user can press 'r' to reset.

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()