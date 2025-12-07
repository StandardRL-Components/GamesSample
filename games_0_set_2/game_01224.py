
# Generated: 2025-08-27T16:26:06.545684
# Source Brief: brief_01224.md
# Brief Index: 1224

        
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
        "Controls: ←→ to move. Press space to push a box in the direction you are facing."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push boxes to clear a path to the green exit door before the time runs out. Plan your moves carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    TILE_SIZE = 40
    GRID_WIDTH = SCREEN_WIDTH // TILE_SIZE  # 16
    GRID_HEIGHT = SCREEN_HEIGHT // TILE_SIZE # 10

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (30, 30, 40)
    COLOR_WALL_MAIN = (60, 60, 70)
    COLOR_WALL_TOP = (80, 80, 90)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_EYE = (255, 255, 255)
    COLOR_BOX_MAIN = (139, 69, 19)
    COLOR_BOX_HIGHLIGHT = (160, 82, 45)
    COLOR_EXIT = (50, 205, 50)
    COLOR_EXIT_GLOW = (150, 255, 150)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TIMER_WARN = (255, 100, 100)

    # Game parameters
    PLAYER_MOVE_COOLDOWN = 5  # frames
    STAGE_TIME_LIMIT = 60.0  # seconds
    VISUAL_LERP_RATE = 0.3 # For smooth movement

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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18)

        self.stage_layouts = self._define_stages()
        
        # This will be initialized in reset()
        self.player_pos = None
        self.player_visual_pos = None
        self.player_facing_dir = 1
        self.player_move_cooldown = 0
        self.box_positions = []
        self.box_visual_positions = []
        self.exit_pos = None
        self.walls = set()
        self.current_stage = 1
        self.time_remaining = 0
        self.last_space_held = False
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()
    
    def _define_stages(self):
        return {
            1: {
                "player_start": (2, 8),
                "exit_pos": (13, 8),
                "boxes": [(8, 8)],
            },
            2: {
                "player_start": (1, 8),
                "exit_pos": (14, 2),
                "boxes": [(5, 8), (5, 7), (5, 6), (9, 4), (9, 3)],
            },
            3: {
                "player_start": (1, 8),
                "exit_pos": (14, 8),
                "boxes": [(4, 8), (4, 7), (8, 8), (8, 7), (8, 6), (12, 8), (12, 7)],
            }
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self._load_stage(self.current_stage)
        
        return self._get_observation(), self._get_info()

    def _load_stage(self, stage_num):
        layout = self.stage_layouts[stage_num]
        
        self.player_pos = pygame.Vector2(layout["player_start"])
        self.player_visual_pos = self.player_pos * self.TILE_SIZE
        self.player_facing_dir = 1
        self.player_move_cooldown = 0

        self.box_positions = [pygame.Vector2(p) for p in layout["boxes"]]
        self.box_visual_positions = [p * self.TILE_SIZE for p in self.box_positions]

        self.exit_pos = pygame.Vector2(layout["exit_pos"])
        
        self.walls = set()
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, -1))
            self.walls.add((x, self.GRID_HEIGHT))
        for y in range(self.GRID_HEIGHT):
            self.walls.add((-1, y))
            self.walls.add((self.GRID_WIDTH, y))

        self.time_remaining = self.STAGE_TIME_LIMIT
        self.last_space_held = False
        self.particles = []

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Cost of time
        terminated = False
        
        self.time_remaining -= 1.0 / self.FPS
        self.steps += 1
        
        # --- Update Cooldowns ---
        if self.player_move_cooldown > 0:
            self.player_move_cooldown -= 1

        # --- Handle Input and Game Logic ---
        if self.player_move_cooldown == 0:
            move_dir = 0
            if movement == 3:  # Left
                move_dir = -1
            elif movement == 4: # Right
                move_dir = 1
            
            if move_dir != 0:
                self.player_facing_dir = move_dir
                target_pos = self.player_pos + pygame.Vector2(move_dir, 0)
                if self._is_pos_free(target_pos):
                    self.player_pos = target_pos
                    self.player_move_cooldown = self.PLAYER_MOVE_COOLDOWN
                    # sfx: player_step.wav

        space_pressed = space_held and not self.last_space_held
        if space_pressed:
            push_reward = self._handle_push()
            reward += push_reward

        self.last_space_held = space_held
        
        # --- Update Visuals (Interpolation) ---
        self._update_visual_positions()
        self._update_particles()
        
        # --- Check Win/Loss Conditions ---
        if self.player_pos == self.exit_pos:
            reward += 100
            self.score += 100
            # sfx: stage_clear.wav
            if self.current_stage < len(self.stage_layouts):
                self.current_stage += 1
                self._load_stage(self.current_stage)
            else:
                terminated = True # Game won
                self.game_over = True

        if self.time_remaining <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            # sfx: game_over.wav
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_push(self):
        target_pos = self.player_pos + pygame.Vector2(self.player_facing_dir, 0)
        box_idx = self._get_box_at(target_pos)

        if box_idx is not None:
            box_target_pos = target_pos + pygame.Vector2(self.player_facing_dir, 0)
            if self._is_pos_free(box_target_pos):
                box_current_pos = self.box_positions[box_idx]
                dist_before = self._manhattan_distance(box_current_pos, self.exit_pos)
                
                self.box_positions[box_idx] = box_target_pos
                # sfx: box_push.wav
                self._create_dust_particles(box_current_pos * self.TILE_SIZE + pygame.Vector2(self.TILE_SIZE / 2, self.TILE_SIZE))
                
                dist_after = self._manhattan_distance(box_target_pos, self.exit_pos)
                
                if dist_after < dist_before:
                    return 1.0  # Positive reward for pushing box closer to exit
                else:
                    return -1.0 # Negative reward for pushing box away
        return 0.0

    def _is_pos_free(self, pos):
        if tuple(pos) in self.walls:
            return False
        if self._get_box_at(pos) is not None:
            return False
        return True

    def _get_box_at(self, pos):
        for i, box_pos in enumerate(self.box_positions):
            if box_pos == pos:
                return i
        return None

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def _update_visual_positions(self):
        target_player_visual = self.player_pos * self.TILE_SIZE
        self.player_visual_pos.x += (target_player_visual.x - self.player_visual_pos.x) * self.VISUAL_LERP_RATE
        self.player_visual_pos.y += (target_player_visual.y - self.player_visual_pos.y) * self.VISUAL_LERP_RATE

        for i in range(len(self.box_positions)):
            target_box_visual = self.box_positions[i] * self.TILE_SIZE
            self.box_visual_positions[i].x += (target_box_visual.x - self.box_visual_positions[i].x) * self.VISUAL_LERP_RATE
            self.box_visual_positions[i].y += (target_box_visual.y - self.box_visual_positions[i].y) * self.VISUAL_LERP_RATE

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
            "stage": self.current_stage,
            "time_remaining": self.time_remaining
        }

    def _render_game(self):
        self._draw_grid()
        self._draw_walls()
        self._draw_exit()
        self._draw_boxes()
        self._draw_player()
        self._draw_particles()

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_walls(self):
        for x in range(self.GRID_WIDTH):
            self._draw_tile_3d((x, 0), self.COLOR_WALL_MAIN, self.COLOR_WALL_TOP)
            self._draw_tile_3d((x, self.GRID_HEIGHT-1), self.COLOR_WALL_MAIN, self.COLOR_WALL_TOP, floor=True)
        for y in range(1, self.GRID_HEIGHT-1):
            self._draw_tile_3d((0, y), self.COLOR_WALL_MAIN, self.COLOR_WALL_TOP)
            self._draw_tile_3d((self.GRID_WIDTH-1, y), self.COLOR_WALL_MAIN, self.COLOR_WALL_TOP)

    def _draw_exit(self):
        exit_rect = pygame.Rect(self.exit_pos.x * self.TILE_SIZE, self.exit_pos.y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        
        # Glow effect
        glow_radius = int(self.TILE_SIZE * (0.8 + 0.1 * math.sin(pygame.time.get_ticks() * 0.005)))
        pygame.gfxdraw.filled_circle(self.screen, exit_rect.centerx, exit_rect.centery, glow_radius, self.COLOR_EXIT_GLOW + (30,))
        
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        pygame.draw.rect(self.screen, self.COLOR_EXIT_GLOW, exit_rect, 2)

    def _draw_boxes(self):
        for pos in self.box_visual_positions:
            self._draw_tile_3d((pos.x / self.TILE_SIZE, pos.y / self.TILE_SIZE), self.COLOR_BOX_MAIN, self.COLOR_BOX_HIGHLIGHT, is_pixel_coords=True)

    def _draw_player(self):
        px, py = self.player_visual_pos.x, self.player_visual_pos.y
        player_rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
        
        # Simple bobbing animation
        bob = math.sin(pygame.time.get_ticks() * 0.01) * 2
        player_rect.y -= bob

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        
        # Eyes to show direction
        eye_y = player_rect.centery - 5
        if self.player_facing_dir == 1: # Right
            eye1_x = player_rect.centerx + 5
            eye2_x = player_rect.centerx + 12
        else: # Left
            eye1_x = player_rect.centerx - 5
            eye2_x = player_rect.centerx - 12
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYE, (eye1_x, eye_y), 3)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYE, (eye2_x, eye_y), 3)

    def _draw_tile_3d(self, grid_pos, main_color, top_color, is_pixel_coords=False, floor=False):
        if is_pixel_coords:
            px, py = grid_pos[0], grid_pos[1]
        else:
            px, py = grid_pos[0] * self.TILE_SIZE, grid_pos[1] * self.TILE_SIZE
        
        main_rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, main_color, main_rect)
        
        if not floor:
            top_rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE // 5)
            pygame.draw.rect(self.screen, top_color, top_rect)
        
        pygame.draw.rect(self.screen, (0,0,0,50), main_rect, 1) # Outline

    def _render_ui(self):
        # Stage display
        stage_text = self.font_small.render(f"Stage: {self.current_stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Timer display
        time_int = int(max(0, self.time_remaining))
        timer_color = self.COLOR_TEXT if time_int > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_main.render(f"{time_int:02d}", True, timer_color)
        text_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, text_rect)

    def _create_dust_particles(self, pos):
        for _ in range(15):
            vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-0.5, -2))
            lifespan = random.randint(10, 20)
            color = random.choice([(101, 67, 33), (87, 58, 28)])
            size = random.randint(2, 5)
            self.particles.append([pygame.Vector2(pos), vel, lifespan, color, size])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1] # pos += vel
            p[2] -= 1    # lifespan -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _draw_particles(self):
        for pos, vel, life, color, size in self.particles:
            pygame.draw.rect(self.screen, color, (pos.x, pos.y, size, size))

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Use a dictionary to track held keys for smooth MultiDiscrete actions
    key_map = {
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
        pygame.K_RSHIFT: False
    }

    # Pygame window for human play
    pygame.display.set_caption("Box Pusher")
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_map:
                    key_map[event.key] = True
            if event.type == pygame.KEYUP:
                if event.key in key_map:
                    key_map[event.key] = False

        # Construct the action from key states
        movement = 0 # no-op
        if key_map[pygame.K_LEFT]:
            movement = 3
        elif key_map[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if key_map[pygame.K_SPACE] else 0
        shift_held = 1 if key_map[pygame.K_LSHIFT] or key_map[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the human-facing screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Final Stage: {info['stage']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        # Control the frame rate for human play
        env.clock.tick(env.FPS)

    env.close()