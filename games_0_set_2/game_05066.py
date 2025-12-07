
# Generated: 2025-08-28T03:52:46.568997
# Source Brief: brief_05066.md
# Brief Index: 5066

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move your character (red square). "
        "Push the blue blocks onto the green goals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-style puzzle game. Push all blocks to their goals before the timer runs out. "
        "Plan your moves carefully to solve each level!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 18, 11
    TILE_SIZE = 32
    GAME_AREA_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH * TILE_SIZE) // 2
    GAME_AREA_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT * TILE_SIZE) // 2 + 20

    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_HIGHLIGHT = (255, 150, 150)
    COLOR_BLOCK = (50, 100, 255)
    COLOR_BLOCK_SHADOW = (30, 60, 180)
    COLOR_BLOCK_ON_GOAL = (100, 200, 255)
    COLOR_GOAL = (50, 150, 50)
    COLOR_GOAL_OCCUPIED = (80, 200, 80)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)
    COLOR_PARTICLE = (255, 220, 50)

    LEVEL_TIME = 60.0
    MOVE_COOLDOWN_FRAMES = 6  # ~5 moves per second at 30fps
    FPS = 30

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)

        # Etc...        
        self.player_pos = np.array([0, 0])
        self.blocks = []
        self.goals = []
        self.visual_player_pos = np.array([0.0, 0.0])
        self.visual_blocks_pos = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.LEVEL_TIME
        self.current_level = 1
        self.moves = 0
        self.move_cooldown = 0
        self.particles = []
        self.solved_blocks = set()
        self.level_complete_timer = 0
        self.level_start_timer = 0

        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def _setup_level(self):
        self.timer = self.LEVEL_TIME
        self.moves = 0
        self.move_cooldown = 0
        self.particles = []
        self.solved_blocks = set()
        self.level_start_timer = self.FPS * 2 # 2 second display
        
        level_data = self._get_level_data(self.current_level)
        self.player_pos = np.array(level_data["player"])
        self.blocks = [np.array(p) for p in level_data["blocks"]]
        self.goals = [np.array(p) for p in level_data["goals"]]

        self.visual_player_pos = self.player_pos.astype(float) * self.TILE_SIZE
        self.visual_blocks_pos = [b.astype(float) * self.TILE_SIZE for b in self.blocks]

    def _get_level_data(self, level):
        if level == 1:
            return {
                "player": [3, 5],
                "blocks": [[5, 5], [9, 3]],
                "goals": [[7, 5], [9, 7]],
            }
        elif level == 2:
            return {
                "player": [2, 5],
                "blocks": [[4, 3], [4, 7], [8, 5], [12, 5]],
                "goals": [[14, 3], [14, 7], [10, 3], [10, 7]],
            }
        elif level == 3:
            return {
                "player": [1, 5],
                "blocks": [[4, 2], [4, 8], [8, 4], [8, 6], [12, 2], [12, 8]],
                "goals": [[15, 1], [15, 9], [6, 1], [6, 9], [10, 5], [10, 6]],
            }
        return self._get_level_data(1) # Default to level 1 if invalid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state, for example:
        self.current_level = 1
        self.score = 0
        self.game_over = False
        self.steps = 0
        self._setup_level()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Update game logic
        self.steps += 1
        reward = -0.01 # Small time penalty

        if self.level_complete_timer > 0:
            self.level_complete_timer -= 1
            if self.level_complete_timer == 0:
                self.current_level += 1
                if self.current_level > 3:
                    self.game_over = True
                    return self._get_observation(), 0, True, False, self._get_info()
                else:
                    self._setup_level()
            self._update_visuals()
            return self._get_observation(), 0, False, False, self._get_info()

        if self.level_start_timer > 0:
            self.level_start_timer -= 1
            self._update_visuals()
            return self._get_observation(), 0, False, False, self._get_info()

        self.timer -= 1.0 / self.FPS
        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        self._handle_movement(movement)
        
        self._update_visuals()

        newly_solved, all_solved = self._check_block_status()
        for _ in newly_solved:
            reward += 10.0 # Reward for placing a block
            self.score += 10

        terminated = self._check_termination(all_solved)
        
        if all_solved and not terminated: # Level complete, but not final level
            reward += 100.0
            self.score += 100
            self.level_complete_timer = self.FPS * 2

        if self.timer <= 0:
            reward -= 100.0
            self.score -= 100
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 0 or self.move_cooldown > 0:
            return

        self.moves += 1
        self.move_cooldown = self.MOVE_COOLDOWN_FRAMES

        direction = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}.get(movement, [0, 0])
        d_pos = np.array(direction)
        
        target_pos = self.player_pos + d_pos
        
        # Boundary check
        if not (0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT):
            return

        # Block collision check
        block_idx = self._get_block_at(target_pos)
        if block_idx is not None:
            block_target_pos = target_pos + d_pos
            # Boundary check for block
            if not (0 <= block_target_pos[0] < self.GRID_WIDTH and 0 <= block_target_pos[1] < self.GRID_HEIGHT):
                return
            # Check if block is blocked by another block
            if self._get_block_at(block_target_pos) is not None:
                return
            
            # Push block
            self.blocks[block_idx] = block_target_pos
            self.player_pos = target_pos
            # Sound: block_push.wav
        else:
            # Move player
            self.player_pos = target_pos
            # Sound: step.wav

    def _check_termination(self, all_solved):
        if self.timer <= 0:
            self.game_over = True
            return True
        if all_solved and self.current_level == 3:
            self.game_over = True
            return True
        return False

    def _get_block_at(self, pos):
        for i, block_pos in enumerate(self.blocks):
            if np.array_equal(block_pos, pos):
                return i
        return None

    def _check_block_status(self):
        current_solved = set()
        for i, block_pos in enumerate(self.blocks):
            if np.array_equal(block_pos, self.goals[i]):
                current_solved.add(i)

        newly_solved = current_solved - self.solved_blocks
        if newly_solved:
            # Sound: success.wav
            for i in newly_solved:
                goal_pos = self.goals[i]
                px, py = self._grid_to_pixel(goal_pos)
                self._create_particles(px + self.TILE_SIZE // 2, py + self.TILE_SIZE // 2)

        self.solved_blocks = current_solved
        all_solved = len(self.solved_blocks) == len(self.blocks)
        return newly_solved, all_solved

    def _update_visuals(self):
        # Interpolate player position
        target_player_px = self.player_pos.astype(float) * self.TILE_SIZE
        self.visual_player_pos += (target_player_px - self.visual_player_pos) * 0.4

        # Interpolate block positions
        for i in range(len(self.blocks)):
            target_block_px = self.blocks[i].astype(float) * self.TILE_SIZE
            self.visual_blocks_pos[i] += (target_block_px - self.visual_blocks_pos[i]) * 0.4
            
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1

    def _create_particles(self, x, y, count=20):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': random.randint(15, 30),
                'size': random.uniform(2, 5)
            })

    def _grid_to_pixel(self, grid_pos):
        return (
            grid_pos[0] * self.TILE_SIZE + self.GAME_AREA_X_OFFSET,
            grid_pos[1] * self.TILE_SIZE + self.GAME_AREA_Y_OFFSET
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.TILE_SIZE + self.GAME_AREA_X_OFFSET
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GAME_AREA_Y_OFFSET), (px, self.GAME_AREA_Y_OFFSET + self.GRID_HEIGHT * self.TILE_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.TILE_SIZE + self.GAME_AREA_Y_OFFSET
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GAME_AREA_X_OFFSET, py), (self.GAME_AREA_X_OFFSET + self.GRID_WIDTH * self.TILE_SIZE, py))

        # Draw goals
        for i, goal_pos in enumerate(self.goals):
            px, py = self._grid_to_pixel(goal_pos)
            color = self.COLOR_GOAL_OCCUPIED if i in self.solved_blocks else self.COLOR_GOAL
            pygame.draw.rect(self.screen, color, (px, py, self.TILE_SIZE, self.TILE_SIZE))

        # Draw blocks
        for i, v_pos in enumerate(self.visual_blocks_pos):
            px, py = int(v_pos[0] + self.GAME_AREA_X_OFFSET), int(v_pos[1] + self.GAME_AREA_Y_OFFSET)
            is_on_goal = i in self.solved_blocks
            color = self.COLOR_BLOCK_ON_GOAL if is_on_goal else self.COLOR_BLOCK
            shadow_color = self.COLOR_BLOCK_SHADOW

            pygame.draw.rect(self.screen, shadow_color, (px + 3, py + 3, self.TILE_SIZE - 6, self.TILE_SIZE - 6))
            pygame.draw.rect(self.screen, color, (px, py, self.TILE_SIZE - 6, self.TILE_SIZE - 6))
            
        # Draw player
        px, py = int(self.visual_player_pos[0] + self.GAME_AREA_X_OFFSET), int(self.visual_player_pos[1] + self.GAME_AREA_Y_OFFSET)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px + 2, py + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_HIGHLIGHT, (px + 4, py + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8))

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(p['x']), int(p['y'])), int(p['size'] * (p['life'] / 30.0)))
            
    def _render_text(self, text, font, x, y, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        self._render_text(f"Level: {self.current_level}", self.font_medium, 20, 10)
        self._render_text(f"Moves: {self.moves}", self.font_medium, 200, 10)
        self._render_text(f"Time: {max(0, self.timer):.1f}", self.font_medium, 360, 10)
        self._render_text(f"Score: {self.score}", self.font_medium, 520, 10)

        if self.level_start_timer > 0:
            alpha = min(255, int(512 * (self.level_start_timer / (self.FPS*2))))
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            self._render_text_centered_alpha(f"LEVEL {self.current_level}", self.font_large, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20, alpha, overlay)
            self._render_text_centered_alpha("GET READY!", self.font_medium, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20, alpha, overlay)
            self.screen.blit(overlay, (0,0))
            
        if self.level_complete_timer > 0 and self.current_level <= 3:
            alpha = min(255, int(512 * (self.level_complete_timer / (self.FPS*2))))
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            msg = "LEVEL COMPLETE!"
            if self.current_level == 3 and self.game_over: msg = "YOU WIN!"
            self._render_text_centered_alpha(msg, self.font_large, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, alpha, overlay)
            self.screen.blit(overlay, (0,0))

        if self.game_over and self.timer <= 0:
            self._render_text("TIME UP!", self.font_large, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, center=True)

    def _render_text_centered_alpha(self, text, font, x, y, alpha, surface):
        text_surf = font.render(text, True, self.COLOR_TEXT)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf.set_alpha(alpha)
        shadow_surf.set_alpha(alpha)
        text_rect = text_surf.get_rect(center=(x,y))
        shadow_rect = shadow_surf.get_rect(center=(x+2,y+2))
        surface.blit(shadow_surf, shadow_rect)
        surface.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level,
            "moves": self.moves,
            "timer": self.timer,
        }

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

if __name__ == '__main__':
    # This block allows you to play the game directly using pygame
    import os
    # To run headlessly, uncomment the next line
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # To display the game, we need a visible pygame window
    pygame.display.set_caption("Block Pusher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Start with no-op action
    action = env.action_space.sample()
    action.fill(0) 

    print("\n" + "="*30)
    print(f"GAME: Block Pusher")
    print(f"INFO: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
        
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move_action = 0 # No-op
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = np.array([move_action, space_action, shift_action])
        # --- End Human Controls ---

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        # Pygame uses (width, height), numpy uses (height, width)
        # The obs is (H, W, C), so we need to transpose for pygame's surfarray
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            
        env.clock.tick(env.FPS)
        
    env.close()