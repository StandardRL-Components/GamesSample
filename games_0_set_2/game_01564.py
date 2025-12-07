
# Generated: 2025-08-27T17:33:17.220292
# Source Brief: brief_01564.md
# Brief Index: 1564

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A minimalist puzzle game where the player pushes colored blocks onto their
    matching target locations on a grid. The player has a limited number of moves.
    All non-locked blocks are pushed simultaneously in the chosen direction until
    they hit a wall or another block.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to push all blocks in a direction. "
        "Solve the puzzle before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Slide colored blocks into their matching "
        "target zones. Each push moves all unlocked blocks simultaneously. "
        "Plan your moves carefully to solve the puzzle in the fewest steps."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    CELL_SIZE = 40
    NUM_BLOCKS = 15
    MAX_MOVES = 25
    SHUFFLE_STEPS = 30 # How many random moves to make to generate a puzzle

    # --- Colors ---
    COLOR_BG = (25, 28, 32)
    COLOR_GRID = (45, 48, 52)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    BLOCK_COLORS = [
        (230, 57, 70), (241, 122, 5), (252, 191, 73), (255, 236, 179),
        (168, 218, 220), (69, 123, 157), (29, 53, 87), (106, 76, 147),
        (199, 125, 255), (255, 154, 201), (100, 181, 246), (0, 200, 83),
        (124, 179, 66), (255, 238, 88), (213, 0, 0)
    ]

    # --- Rewards ---
    REWARD_PER_MOVE = -0.1
    REWARD_BLOCK_ON_TARGET = 1.0
    REWARD_WIN = 100.0
    REWARD_LOSS = -10.0

    # --- Animation ---
    ANIMATION_FRAMES = 6 # 6 frames at 30fps = 0.2s animation

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
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.blocks = []
        self.targets = []
        self.visual_blocks = []
        self.effects = []
        self.moves_remaining = 0
        self.score = 0
        self.game_over = False
        self.game_state = "WAITING" # WAITING, ANIMATING
        self.animation_timer = 0
        self.steps = 0

        # Validate implementation after full initialization
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.game_state = "WAITING"
        self.animation_timer = 0
        self.effects = []
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        # 1. Generate solved state
        all_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_cells)
        
        solved_positions = all_cells[:self.NUM_BLOCKS]
        
        self.blocks = []
        self.targets = []
        
        colors = self.BLOCK_COLORS[:self.NUM_BLOCKS]
        self.np_random.shuffle(colors)

        for i in range(self.NUM_BLOCKS):
            pos = solved_positions[i]
            color = colors[i]
            self.targets.append({'pos': pos, 'color': color, 'id': i})
            self.blocks.append({'pos': pos, 'color': color, 'id': i, 'locked': True})

        # 2. Shuffle by applying random moves from the solved state
        for _ in range(self.SHUFFLE_STEPS):
            for block in self.blocks:
                block['locked'] = False
            
            random_move = self.np_random.integers(1, 5) # 1-4 for directions
            self._apply_push(random_move, is_shuffle=True)
            
            for block in self.blocks:
                if self.visual_blocks[block['id']] is not None:
                    block['pos'] = self.visual_blocks[block['id']]['end_pos']

        # 3. Finalize setup: unlock all blocks and check for accidental solves
        for block in self.blocks:
            block['locked'] = False
        
        self._check_and_lock_solved_blocks()

    def _check_and_lock_solved_blocks(self):
        for block in self.blocks:
            if not block['locked']:
                target = self.targets[block['id']]
                if block['pos'] == target['pos']:
                    block['locked'] = True
    
    def step(self, action):
        movement = action[0]
        self.steps += 1
        reward = 0

        if self.game_state == "ANIMATING":
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                self.game_state = "WAITING"
                for block in self.blocks:
                    if not block['locked']:
                        block['pos'] = self.visual_blocks[block['id']]['end_pos']
                
                if self._check_win_condition():
                    self.game_over = True
                    reward += self.REWARD_WIN
                    # sound: win_fanfare.wav
            
            return self._get_observation(), reward, self.game_over, False, self._get_info()

        if self.game_over:
            return self._get_observation(), 0, self.game_over, False, self._get_info()

        if movement != 0: # A valid move action
            self.moves_remaining -= 1
            reward += self.REWARD_PER_MOVE
            # sound: push.wav
            
            move_reward = self._apply_push(movement)
            reward += move_reward
            self.score += move_reward

            self.game_state = "ANIMATING"
            self.animation_timer = self.ANIMATION_FRAMES
            
            if self.moves_remaining <= 0 and not self._check_win_condition():
                self.game_over = True
                reward += self.REWARD_LOSS
                # sound: lose_buzzer.wav

        return self._get_observation(), reward, self.game_over, False, self._get_info()
    
    def _apply_push(self, direction, is_shuffle=False):
        if direction == 1: dx, dy, sort_key, rev = 0, -1, 1, False
        elif direction == 2: dx, dy, sort_key, rev = 0, 1, 1, True
        elif direction == 3: dx, dy, sort_key, rev = -1, 0, 0, False
        else: dx, dy, sort_key, rev = 1, 0, 0, True

        movable_blocks = sorted([b for b in self.blocks if not b['locked']], key=lambda b: b['pos'][sort_key], reverse=rev)
        occupied = {b['pos']: True for b in self.blocks}
        move_reward = 0
        self.visual_blocks = [None] * self.NUM_BLOCKS
        
        for block in self.blocks:
            if block['locked']:
                 self.visual_blocks[block['id']] = {'start_pos': block['pos'], 'end_pos': block['pos'], 'color': block['color']}

        for block in movable_blocks:
            start_pos = block['pos']
            current_pos = start_pos
            
            while True:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT): break
                if occupied.get(next_pos, False): break
                current_pos = next_pos
            
            end_pos = current_pos
            self.visual_blocks[block['id']] = {'start_pos': start_pos, 'end_pos': end_pos, 'color': block['color']}
            
            del occupied[start_pos]
            occupied[end_pos] = True

            target = self.targets[block['id']]
            if end_pos == target['pos']:
                block['locked'] = True
                if not is_shuffle:
                    move_reward += self.REWARD_BLOCK_ON_TARGET
                    # sound: lock_in.wav
                    self.effects.append({'type': 'lock_in', 'pos': end_pos, 'timer': 10, 'max_timer': 10})
        return move_reward

    def _check_win_condition(self):
        return all(b['locked'] for b in self.blocks)

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
            "moves_remaining": self.moves_remaining,
            "game_state": self.game_state,
            "is_animating": self.game_state == "ANIMATING",
            "blocks_locked": sum(1 for b in self.blocks if b['locked'])
        }

    def _render_game(self):
        self._render_grid()
        self._render_targets()
        self._render_blocks()
        self._render_effects()

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

    def _render_targets(self):
        for target in self.targets:
            x, y = target['pos']
            px, py = x * self.CELL_SIZE, y * self.CELL_SIZE
            rect = (px, py, self.CELL_SIZE, self.CELL_SIZE)
            
            target_color = tuple(c * 0.4 for c in target['color'])
            pygame.gfxdraw.box(self.screen, rect, target_color)
            
            pygame.draw.line(self.screen, self.COLOR_GRID, (px + 5, py + 5), (px + self.CELL_SIZE - 6, py + self.CELL_SIZE - 6), 2)
            pygame.draw.line(self.screen, self.COLOR_GRID, (px + self.CELL_SIZE - 6, py + 5), (px + 5, py + self.CELL_SIZE - 6), 2)

    def _render_blocks(self):
        if self.game_state == "ANIMATING":
            t = 1.0 - (self.animation_timer / self.ANIMATION_FRAMES)
            t = 1 - (1 - t) * (1 - t) # Ease out quadratic
        else:
            t = 1.0

        for i in range(len(self.visual_blocks)):
            if self.visual_blocks[i] is None: continue
            
            block_data = self.visual_blocks[i]
            block_state = self.blocks[i]
            
            start_x, start_y = block_data['start_pos']
            end_x, end_y = block_data['end_pos']
            
            interp_px = (start_x * (1 - t) + end_x * t) * self.CELL_SIZE
            interp_py = (start_y * (1 - t) + end_y * t) * self.CELL_SIZE
            
            rect = pygame.Rect(int(interp_px), int(interp_py), self.CELL_SIZE, self.CELL_SIZE)
            padding = 3
            block_rect = rect.inflate(-padding * 2, -padding * 2)
            
            shadow_rect = block_rect.move(2, 3)
            pygame.draw.rect(self.screen, (0,0,0,50), shadow_rect, border_radius=5)
            pygame.draw.rect(self.screen, block_data['color'], block_rect, border_radius=5)
            pygame.draw.rect(self.screen, tuple(min(255, c+30) for c in block_data['color']), block_rect, 2, border_radius=5)
            
            if block_state['locked']:
                self._draw_star(self.screen, block_rect.center, 6, 5, (255, 255, 255))
    
    def _draw_star(self, surface, center, outer_r, inner_r, color):
        points = []
        for i in range(10):
            angle = math.radians(i * 36 - 90) # Align star upwards
            r = outer_r if i % 2 == 0 else inner_r
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            points.append((x, y))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_effects(self):
        for effect in self.effects[:]:
            effect['timer'] -= 1
            if effect['timer'] <= 0:
                self.effects.remove(effect)
                continue
            
            if effect['type'] == 'lock_in':
                progress = 1.0 - (effect['timer'] / effect['max_timer'])
                center_px = (int((effect['pos'][0] + 0.5) * self.CELL_SIZE), int((effect['pos'][1] + 0.5) * self.CELL_SIZE))
                radius = int(progress * self.CELL_SIZE * 0.6)
                alpha = int((1.0 - progress) * 200)
                if radius > 1 and alpha > 0:
                    pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], radius, (255, 255, 255, alpha))

    def _render_ui(self):
        text = f"Moves: {self.moves_remaining}"
        pos = (15, 15)
        self._draw_text(text, self.font_large, pos)
        
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            msg = "PUZZLE SOLVED!" if self._check_win_condition() else "OUT OF MOVES"
            self._draw_text(msg, self.font_large, (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2 - 20), center=True)
            
    def _draw_text(self, text, font, pos, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        text_shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect(center=pos) if center else text_surf.get_rect(topleft=pos)
        shadow_rect = text_rect.move(2, 2)
        self.screen.blit(text_shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        try:
            obs, info = self.reset()
        except Exception as e:
            raise Exception(f"Validation failed: reset() raised an exception: {e}")
            
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    key_to_action = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
    }

    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action[0] = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                if not done and not info.get("is_animating", False):
                    if event.key in key_to_action:
                        action[0] = key_to_action[event.key]
            if event.type == pygame.KEYUP:
                if event.key in key_to_action and action[0] == key_to_action[event.key]:
                     action[0] = 0

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if reward != 0:
                print(f"Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_remaining']}")

        if info.get("is_animating"):
            action[0] = 0
            
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30)

    env.close()