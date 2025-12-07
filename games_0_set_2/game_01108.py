
# Generated: 2025-08-27T16:04:17.606913
# Source Brief: brief_01108.md
# Brief Index: 1108

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press Space to swap with the gem to the right, or Shift to swap with the gem below."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. Create combos and chain reactions to reach the target score of 50 gems within 20 moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Game parameters
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_GEM_TYPES = 5
    GEM_GOAL = 50
    MOVE_LIMIT = 20
    MAX_STEPS = 1000

    # Screen dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Visuals
    CELL_SIZE = 40
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2
    GEM_RADIUS = CELL_SIZE // 2 - 4
    
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_UI_TEXT = (220, 220, 230)
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 255, 80),   # Yellow
        (200, 80, 255),   # Purple
    ]
    SELECTOR_COLOR = (255, 255, 255)

    # Animation speeds (in steps)
    ANIM_SWAP_DURATION = 6
    ANIM_MATCH_DURATION = 8
    ANIM_FALL_DURATION = 10

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.game_state = "INIT"
        self.grid = None
        self.selector_pos = None
        self.moves_left = None
        self.gems_collected = None
        self.game_over = None
        self.animation_state = None
        self.turn_reward = 0
        self.chain_reaction_level = 0
        self.steps = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._create_initial_grid()
        self.selector_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.moves_left = self.MOVE_LIMIT
        self.gems_collected = 0
        self.score = 0
        self.game_over = False
        self.steps = 0
        
        self.game_state = "AWAITING_INPUT"
        self.animation_state = None
        self.turn_reward = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False

        if self.game_state == "AWAITING_INPUT":
            self._handle_input(action)
        elif self.game_state == "ANIMATING_SWAP":
            self._update_swap_animation()
        elif self.game_state == "PROCESSING_MATCHES":
            self._process_matches()
        elif self.game_state == "ANIMATING_MATCH":
            self._update_match_animation()
        elif self.game_state == "PROCESSING_FALL":
            self._process_fall()
        elif self.game_state == "ANIMATING_FALL":
            self._update_fall_animation()

        if self.game_state == "AWAITING_INPUT" and self.turn_reward != 0:
            reward = self.turn_reward
            self.score += reward
            self.turn_reward = 0

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.gems_collected >= self.GEM_GOAL:
                reward += 100 # Win bonus
            else:
                reward += -50 # Loss penalty
            self.score += reward

        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle selector movement ---
        if movement == 1: # Up
            self.selector_pos[1] = max(0, self.selector_pos[1] - 1)
        elif movement == 2: # Down
            self.selector_pos[1] = min(self.GRID_HEIGHT - 1, self.selector_pos[1] + 1)
        elif movement == 3: # Left
            self.selector_pos[0] = max(0, self.selector_pos[0] - 1)
        elif movement == 4: # Right
            self.selector_pos[0] = min(self.GRID_WIDTH - 1, self.selector_pos[0] + 1)

        # --- Handle swap action ---
        swap_target_pos = None
        if space_pressed:
            if self.selector_pos[0] < self.GRID_WIDTH - 1:
                swap_target_pos = [self.selector_pos[0] + 1, self.selector_pos[1]]
        elif shift_pressed:
            if self.selector_pos[1] < self.GRID_HEIGHT - 1:
                swap_target_pos = [self.selector_pos[0], self.selector_pos[1] + 1]

        if swap_target_pos:
            self.moves_left -= 1
            self.chain_reaction_level = 0
            
            p1 = self.selector_pos
            p2 = swap_target_pos
            
            # Perform swap in grid data
            self.grid[p1[1], p1[0]], self.grid[p2[1], p2[0]] = self.grid[p2[1], p2[0]], self.grid[p1[1], p1[0]]
            
            matches = self._find_matches()
            
            # Swap back if no matches
            self.grid[p1[1], p1[0]], self.grid[p2[1], p2[0]] = self.grid[p2[1], p2[0]], self.grid[p1[1], p1[0]]
            
            self.animation_state = {
                "type": "swap",
                "pos1": tuple(p1), "pos2": tuple(p2),
                "is_valid": len(matches) > 0,
                "progress": 0
            }
            self.game_state = "ANIMATING_SWAP"

    def _update_swap_animation(self):
        self.animation_state["progress"] += 1
        if self.animation_state["progress"] >= self.ANIM_SWAP_DURATION:
            p1 = self.animation_state["pos1"]
            p2 = self.animation_state["pos2"]
            
            if self.animation_state["is_valid"]:
                # Finalize swap in grid
                self.grid[p1[1], p1[0]], self.grid[p2[1], p2[0]] = self.grid[p2[1], p2[0]], self.grid[p1[1], p1[0]]
                self.game_state = "PROCESSING_MATCHES"
            else:
                # Invalid swap
                self.turn_reward = -0.1
                self.game_state = "AWAITING_INPUT"
            self.animation_state = None

    def _process_matches(self):
        matches = self._find_matches()
        if not matches:
            self.game_state = "AWAITING_INPUT"
            return
            
        if self.chain_reaction_level > 0:
            self.turn_reward += 5 # Chain reaction bonus
            # sfx: chain_reaction_sound()
        
        self.chain_reaction_level += 1
        
        num_matched = len(matches)
        self.gems_collected += num_matched
        self.turn_reward += num_matched # +1 per gem

        # sfx: match_found_sound()
        
        self.animation_state = {
            "type": "match",
            "gems": [(r, c, self.grid[r, c]) for r, c in matches],
            "progress": 0
        }
        for r, c in matches:
            self.grid[r, c] = 0 # Mark as empty
        
        self.game_state = "ANIMATING_MATCH"

    def _update_match_animation(self):
        self.animation_state["progress"] += 1
        if self.animation_state["progress"] >= self.ANIM_MATCH_DURATION:
            self.animation_state = None
            self.game_state = "PROCESSING_FALL"

    def _process_fall(self):
        falling_gems = []
        for c in range(self.GRID_WIDTH):
            write_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != write_row:
                        gem_type = self.grid[r, c]
                        falling_gems.append({
                            "from": (r, c), "to": (write_row, c), "type": gem_type
                        })
                        self.grid[write_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    write_row -= 1
        
        # Fill new gems at the top
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT):
                if self.grid[r, c] == 0:
                    gem_type = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
                    falling_gems.append({
                        "from": (-1 - r, c), # Start from above the screen
                        "to": (r, c), "type": gem_type
                    })
                    self.grid[r, c] = gem_type
        
        if falling_gems:
            # sfx: gems_falling_sound()
            self.animation_state = {
                "type": "fall",
                "gems": falling_gems,
                "progress": 0
            }
            self.game_state = "ANIMATING_FALL"
        else:
            self.game_state = "PROCESSING_MATCHES" # Check for new matches

    def _update_fall_animation(self):
        self.animation_state["progress"] += 1
        if self.animation_state["progress"] >= self.ANIM_FALL_DURATION:
            self.animation_state = None
            self.game_state = "PROCESSING_MATCHES" # Check for chain reactions

    def _find_matches(self):
        to_remove = set()
        # Horizontal
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                gem_type = self.grid[r, c]
                if gem_type != 0 and gem_type == self.grid[r, c+1] and gem_type == self.grid[r, c+2]:
                    to_remove.add((r, c)); to_remove.add((r, c+1)); to_remove.add((r, c+2))
        # Vertical
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                gem_type = self.grid[r, c]
                if gem_type != 0 and gem_type == self.grid[r+1, c] and gem_type == self.grid[r+2, c]:
                    to_remove.add((r, c)); to_remove.add((r+1, c)); to_remove.add((r+2, c))
        return to_remove

    def _create_initial_grid(self):
        while True:
            grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            original_grid = self.grid
            self.grid = grid
            if not self._find_matches():
                return grid
            self.grid = original_grid

    def _check_termination(self):
        return self.moves_left <= 0 or self.gems_collected >= self.GEM_GOAL or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_gems()
        if not self.game_over:
            self._render_selector()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_and_gems(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.GRID_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT * self.CELL_SIZE))

        # Prepare list of gems to draw, considering animations
        gems_to_draw = []
        animated_gems = set()

        if self.animation_state:
            progress_ratio = self.animation_state["progress"] / {
                "swap": self.ANIM_SWAP_DURATION,
                "match": self.ANIM_MATCH_DURATION,
                "fall": self.ANIM_FALL_DURATION
            }.get(self.animation_state["type"], 1)

            if self.animation_state["type"] == "swap":
                p1 = self.animation_state["pos1"]
                p2 = self.animation_state["pos2"]
                is_valid = self.animation_state["is_valid"]
                
                # Determine which gem is which, pre-swap
                gem_type1, gem_type2 = (self.grid[p1[1], p1[0]], self.grid[p2[1], p2[0]]) if is_valid else (self.grid[p2[1], p2[0]], self.grid[p1[1], p1[0]])

                pos1_x, pos1_y = self._grid_to_pixel(p1[0], p1[1])
                pos2_x, pos2_y = self._grid_to_pixel(p2[0], p2[1])

                # Interpolate positions
                interp_x1 = int(pos1_x + (pos2_x - pos1_x) * progress_ratio)
                interp_y1 = int(pos1_y + (pos2_y - pos1_y) * progress_ratio)
                interp_x2 = int(pos2_x + (pos1_x - pos2_x) * progress_ratio)
                interp_y2 = int(pos2_y + (pos1_y - pos2_y) * progress_ratio)

                if not is_valid: # Animate swap and back
                    back_ratio = max(0, progress_ratio * 2 - 1)
                    interp_x1 = int(interp_x1 - (pos2_x - pos1_x) * back_ratio)
                    interp_y1 = int(interp_y1 - (pos2_y - pos1_y) * back_ratio)
                    interp_x2 = int(interp_x2 - (pos1_x - pos2_x) * back_ratio)
                    interp_y2 = int(interp_y2 - (pos1_y - pos2_y) * back_ratio)
                
                gems_to_draw.append({'x': interp_x1, 'y': interp_y1, 'type': gem_type1, 'scale': 1.0})
                gems_to_draw.append({'x': interp_x2, 'y': interp_y2, 'type': gem_type2, 'scale': 1.0})
                animated_gems.add(p1)
                animated_gems.add(p2)

            elif self.animation_state["type"] == "match":
                scale = 1.0 - progress_ratio
                for r, c, gem_type in self.animation_state["gems"]:
                    x, y = self._grid_to_pixel(c, r)
                    gems_to_draw.append({'x': x, 'y': y, 'type': gem_type, 'scale': scale})
                    animated_gems.add((r, c))

            elif self.animation_state["type"] == "fall":
                for gem_info in self.animation_state["gems"]:
                    from_r, from_c = gem_info["from"]
                    to_r, to_c = gem_info["to"]
                    from_x, from_y = self._grid_to_pixel(from_c, from_r)
                    to_x, to_y = self._grid_to_pixel(to_c, to_r)
                    
                    interp_x = int(from_x + (to_x - from_x) * progress_ratio)
                    interp_y = int(from_y + (to_y - from_y) * progress_ratio)
                    
                    gems_to_draw.append({'x': interp_x, 'y': interp_y, 'type': gem_info["type"], 'scale': 1.0})
                    animated_gems.add(gem_info["to"])


        # Draw static gems
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r, c) not in animated_gems and self.grid[r, c] != 0:
                    x, y = self._grid_to_pixel(c, r)
                    gems_to_draw.append({'x': x, 'y': y, 'type': self.grid[r, c], 'scale': 1.0})

        # Render all gems
        for gem in gems_to_draw:
            self._draw_gem(gem['x'], gem['y'], gem['type'], gem['scale'])

    def _render_selector(self):
        if self.game_state != "AWAITING_INPUT":
            return
            
        x, y = self._grid_to_pixel(self.selector_pos[0], self.selector_pos[1])
        
        # Pulsing effect
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        size = self.CELL_SIZE - 2 + pulse * 4
        alpha = 150 + pulse * 105
        
        rect = pygame.Rect(x - size/2, y - size/2, size, size)
        
        # Create a temporary surface for transparency
        shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, (*self.SELECTOR_COLOR, alpha), shape_surf.get_rect(), border_radius=8, width=3)
        self.screen.blit(shape_surf, rect.topleft)

    def _render_ui(self):
        # Gems collected
        gem_text = f"Gems: {self.gems_collected} / {self.GEM_GOAL}"
        gem_surf = self.font_large.render(gem_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(gem_surf, (20, 20))
        
        # Moves left
        moves_text = f"Moves: {self.moves_left}"
        moves_surf = self.font_large.render(moves_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_surf, (self.SCREEN_WIDTH - moves_surf.get_width() - 20, 20))

        # Game Over message
        if self.game_over:
            if self.gems_collected >= self.GEM_GOAL:
                msg = "YOU WIN!"
                color = self.GEM_COLORS[1]
            else:
                msg = "GAME OVER"
                color = self.GEM_COLORS[0]
            
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            
            # Draw a semi-transparent background for the text
            bg_rect = msg_rect.inflate(20, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 180))
            self.screen.blit(bg_surf, bg_rect.topleft)
            self.screen.blit(msg_surf, msg_rect)


    def _draw_gem(self, x, y, gem_type, scale=1.0):
        if gem_type == 0 or scale <= 0: return
        radius = int(self.GEM_RADIUS * scale)
        if radius <= 0: return
        
        color = self.GEM_COLORS[gem_type - 1]
        
        # Use gfxdraw for antialiasing
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        
        # Add a subtle highlight
        highlight_color = tuple(min(255, c + 60) for c in color)
        pygame.gfxdraw.aacircle(self.screen, x, y, int(radius * 0.8), highlight_color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius * 0.8), highlight_color)
        
        inner_color = tuple(min(255, c + 30) for c in color)
        pygame.gfxdraw.aacircle(self.screen, x, y, int(radius * 0.5), inner_color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius * 0.5), inner_color)


    def _grid_to_pixel(self, c, r):
        x = self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "moves_left": self.moves_left,
            "game_state": self.game_state
        }
        
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a separate window for rendering
    pygame.display.set_caption("Gem Swap")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    action = [0, 0, 0] # no-op, released, released
    
    print("\n" + "="*30)
    print("Gem Swap - Manual Control")
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        # --- Human Controls ---
        # Reset action at the start of each frame
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # This game is turn-based, so we only process keys when awaiting input
        if info.get("game_state") == "AWAITING_INPUT":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, State: {info['game_state']}")

        # --- Rendering ---
        # Transpose observation back to pygame's (W, H, C) format
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # In a turn-based game, we wait for input, so a low FPS is fine
        env.clock.tick(30)
        
    print("\nGame Over!")
    print(f"Final Score: {info['score']:.2f}")
    print(f"Gems Collected: {info['gems_collected']}")
    
    # Wait a bit before closing
    pygame.time.wait(3000)
    env.close()