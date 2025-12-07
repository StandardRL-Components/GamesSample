
# Generated: 2025-08-27T19:11:15.791213
# Source Brief: brief_02066.md
# Brief Index: 2066

        
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
        "Controls: Arrow keys to jump between grid cells. Collect orbs to score points before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade puzzle game. Control a ninja to collect valuable orbs on a grid. Green orbs give points, red orbs take them away, and blue orbs add time. Reach 100 points across three stages to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    GRID_ROWS, GRID_COLS = 10, 10
    GRID_CELL_SIZE = 32
    GRID_WIDTH = GRID_COLS * GRID_CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * GRID_CELL_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_PLAYER = (255, 80, 120)
    COLOR_PLAYER_TRAIL = (255, 80, 120, 150)
    COLOR_ORB_GREEN = (80, 255, 150)
    COLOR_ORB_RED = (255, 70, 70)
    COLOR_ORB_BLUE = (100, 180, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SCORE = (255, 215, 0)
    COLOR_TIME = (100, 180, 255)
    
    # Game parameters
    INITIAL_TIME = 60.0
    STAGE_GOALS = [30, 70, 100]
    JUMP_DURATION_FRAMES = 6 # 0.2s at 30fps
    JUMP_HEIGHT = 20
    ORB_MOVE_INTERVAL_SECONDS = 2


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_orb = pygame.font.Font(None, 20)
        self.font_msg = pygame.font.Font(None, 60)
        
        # Initialize state variables
        self.np_random = None
        self.stage = 0
        self.score = 0
        self.time_remaining = 0
        self.player_pos = [0, 0]
        self.player_visual_pos = [0, 0]
        self.player_anim_start_pos = [0, 0]
        self.player_anim_target_pos = [0, 0]
        self.player_anim_progress = 0
        self.is_jumping = False
        self.orbs = []
        self.particles = []
        self.game_over = False
        self.win = False
        self.steps = 0
        self.orb_move_timer = 0
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.stage = 1
        self.score = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        
        self._setup_stage()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Initializes the state for the current stage."""
        self.time_remaining = self.INITIAL_TIME
        self.player_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.player_visual_pos = self._grid_to_screen(self.player_pos)
        self.is_jumping = False
        self.player_anim_progress = 0
        self.orbs = []
        self.particles = []
        self.orb_move_timer = 0
        
        self._spawn_orbs()

    def _spawn_orbs(self):
        """Spawns orbs based on the current stage."""
        self.orbs.clear()
        
        num_green = 8
        num_red = 3 + (self.stage - 1) * 3
        num_blue = 3
        
        available_cells = [(r, c) for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS)]
        available_cells.remove(tuple(self.player_pos))
        self.np_random.shuffle(available_cells)

        # Spawn Green Orbs
        for _ in range(num_green):
            if not available_cells: break
            pos = available_cells.pop()
            self.orbs.append(self._create_orb('green', pos))
        
        # Spawn Red Orbs
        for _ in range(num_red):
            if not available_cells: break
            pos = available_cells.pop()
            self.orbs.append(self._create_orb('red', pos))

        # Spawn Blue Orbs
        for _ in range(num_blue):
            if not available_cells: break
            pos = available_cells.pop()
            self.orbs.append(self._create_orb('blue', pos))
            
        # Ensure at least one positive orb exists
        if not any(o['type'] == 'green' for o in self.orbs):
            if not available_cells:
                 # Extremely unlikely case: grid is full, replace a red orb
                 red_orbs = [o for o in self.orbs if o['type'] == 'red']
                 if red_orbs:
                     self.orbs.remove(red_orbs[0])
                     self.orbs.append(self._create_orb('green', red_orbs[0]['pos']))
            else:
                 pos = available_cells.pop()
                 self.orbs.append(self._create_orb('green', pos))


    def _create_orb(self, orb_type, pos):
        """Helper to create an orb dictionary."""
        is_moving = self.stage == 3 and self.np_random.random() < 0.25 # 25% chance to be a moving orb in stage 3
        value = 0
        if orb_type == 'green':
            value = self.np_random.integers(1, 11)
        elif orb_type == 'red':
            value = self.np_random.integers(1, 6)
        elif orb_type == 'blue':
            value = 5 # +5 seconds
            
        return {
            'pos': list(pos), 
            'visual_pos': self._grid_to_screen(pos),
            'type': orb_type, 
            'value': value,
            'moving': is_moving,
            'is_animating': False,
            'anim_progress': 0,
            'anim_start_pos': self._grid_to_screen(pos),
            'anim_target_pos': self._grid_to_screen(pos),
        }
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0.0

        # --- Update Time ---
        self.time_remaining -= 1 / self.FPS
        
        # --- Handle Input ---
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        if movement != 0 and not self.is_jumping:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            target_pos = [self.player_pos[0] + dy, self.player_pos[1] + dx]

            if 0 <= target_pos[0] < self.GRID_ROWS and 0 <= target_pos[1] < self.GRID_COLS:
                self.is_jumping = True
                self.player_anim_progress = 0
                self.player_anim_start_pos = self._grid_to_screen(self.player_pos)
                self.player_anim_target_pos = self._grid_to_screen(target_pos)
                # Logical position updates upon landing
                self.logical_target_pos = target_pos
                # Sound: Jump

        # --- Update Game Logic and Animations ---
        self._update_player_animation()
        reward_from_landing = self._update_landing_consequences()
        reward += reward_from_landing
        
        self._update_orb_movement()
        self._update_particles()
        
        # --- Check for Stage Completion ---
        if self.score >= self.STAGE_GOALS[self.stage - 1]:
            reward += 5.0 # Stage complete bonus
            reward += self.time_remaining * 0.1 # Time bonus
            
            self.stage += 1
            if self.stage > 3:
                self.win = True
                self.game_over = True
                reward += 50.0 # Win game bonus
                # Sound: Game Win
            else:
                self._setup_stage()
                # Sound: Stage Clear

        # --- Check Termination ---
        terminated = False
        if self.time_remaining <= 0:
            self.game_over = True
            if not self.win:
                reward = -10.0 # Penalty for losing
                # Sound: Game Over
        
        if self.game_over:
            terminated = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_player_animation(self):
        if self.is_jumping:
            self.player_anim_progress += 1 / self.JUMP_DURATION_FRAMES
            if self.player_anim_progress >= 1.0:
                self.player_anim_progress = 1.0
                self.is_jumping = False
                self.player_pos = self.logical_target_pos
                # Sound: Land
            
            t = self.player_anim_progress
            self.player_visual_pos[0] = self._lerp(self.player_anim_start_pos[0], self.player_anim_target_pos[0], t)
            self.player_visual_pos[1] = self._lerp(self.player_anim_start_pos[1], self.player_anim_target_pos[1], t)
            # Add hop effect
            hop = math.sin(t * math.pi) * self.JUMP_HEIGHT
            self.player_visual_pos[1] -= hop

    def _update_landing_consequences(self):
        """Checks for orb collection after a jump animation finishes."""
        if self.is_jumping or self.player_anim_progress != 1.0:
            return 0.0
            
        reward = 0.0
        collided_orb = None
        for orb in self.orbs:
            if orb['pos'] == self.player_pos:
                collided_orb = orb
                break

        if collided_orb:
            # Sound: Collect Orb
            if collided_orb['type'] == 'green':
                self.score += collided_orb['value']
                reward += 1.0
                self._create_particles(self._grid_to_screen(self.player_pos), self.COLOR_ORB_GREEN, 20)
            elif collided_orb['type'] == 'red':
                self.score -= collided_orb['value']
                self.score = max(0, self.score) # Score cannot be negative
                reward -= 1.0
                self._create_particles(self._grid_to_screen(self.player_pos), self.COLOR_ORB_RED, 20)
            elif collided_orb['type'] == 'blue':
                self.time_remaining += collided_orb['value']
                self.time_remaining = min(self.INITIAL_TIME, self.time_remaining) # Cap time
                self._create_particles(self._grid_to_screen(self.player_pos), self.COLOR_ORB_BLUE, 20)
            
            self.orbs.remove(collided_orb)
            
            # Anti-softlock: ensure a positive orb is available
            if not any(o['type'] == 'green' for o in self.orbs):
                self._spawn_orbs() # Respawn all orbs to be safe

        self.player_anim_progress = 0 # Reset for next jump
        return reward

    def _update_orb_movement(self):
        if self.stage != 3: return
        
        self.orb_move_timer += 1 / self.FPS
        if self.orb_move_timer >= self.ORB_MOVE_INTERVAL_SECONDS:
            self.orb_move_timer = 0
            
            moving_orbs = [o for o in self.orbs if o['moving'] and not o['is_animating']]
            if moving_orbs:
                orb_to_move = self.np_random.choice(moving_orbs)
                
                # Find valid adjacent empty cell
                r, c = orb_to_move['pos']
                possible_moves = []
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                        is_occupied = any(o['pos'] == [nr, nc] for o in self.orbs) or self.player_pos == [nr, nc]
                        if not is_occupied:
                            possible_moves.append([nr, nc])
                
                if possible_moves:
                    target_pos = list(self.np_random.choice(np.array(possible_moves)))
                    orb_to_move['is_animating'] = True
                    orb_to_move['anim_progress'] = 0
                    orb_to_move['anim_start_pos'] = orb_to_move['visual_pos']
                    orb_to_move['anim_target_pos'] = self._grid_to_screen(target_pos)
                    orb_to_move['logical_target_pos'] = target_pos
                    
        # Update animation for any moving orbs
        for orb in self.orbs:
            if orb['is_animating']:
                orb['anim_progress'] += 1 / (self.JUMP_DURATION_FRAMES * 2) # Slower than player
                if orb['anim_progress'] >= 1.0:
                    orb['anim_progress'] = 1.0
                    orb['is_animating'] = False
                    orb['pos'] = orb['logical_target_pos']
                
                t = orb['anim_progress']
                orb['visual_pos'][0] = self._lerp(orb['anim_start_pos'][0], orb['anim_target_pos'][0], t)
                orb['visual_pos'][1] = self._lerp(orb['anim_start_pos'][1], orb['anim_target_pos'][1], t)


    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30) # frames
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

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
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time": self.time_remaining
        }

    def _render_game(self):
        self._render_grid()
        self._render_orbs()
        self._render_particles()
        self._render_player()

    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.GRID_CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.GRID_CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT), 1)

    def _render_orbs(self):
        for orb in self.orbs:
            pos = [int(p) for p in orb['visual_pos']]
            color = self.COLOR_ORB_GREEN if orb['type'] == 'green' else self.COLOR_ORB_RED if orb['type'] == 'red' else self.COLOR_ORB_BLUE
            
            radius = self.GRID_CELL_SIZE // 2 - 4
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            
            text_surf = self.font_orb.render(str(orb['value']), True, self.COLOR_BG)
            text_rect = text_surf.get_rect(center=pos)
            self.screen.blit(text_surf, text_rect)

    def _render_player(self):
        pos = [int(p) for p in self.player_visual_pos]
        size = self.GRID_CELL_SIZE // 2 - 2
        player_rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
        
        # Simple ninja shape (rounded rect)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Eye slit
        eye_rect = pygame.Rect(player_rect.left, player_rect.centery - 2, player_rect.width, 4)
        pygame.draw.rect(self.screen, self.COLOR_BG, eye_rect)

    def _render_particles(self):
        for p in self.particles:
            pos = [int(p['pos'][0]), int(p['pos'][1])]
            size = max(1, int(p['lifespan'] / 6))
            pygame.draw.circle(self.screen, p['color'], pos, size)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_SCORE)
        self.screen.blit(score_surf, (20, 10))

        # Time
        time_text = f"TIME: {max(0, int(self.time_remaining))}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TIME)
        time_rect = time_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(time_surf, time_rect)

        # Stage
        stage_text = f"STAGE {self.stage}"
        stage_surf = self.font_ui.render(stage_text, True, self.COLOR_TEXT)
        stage_rect = stage_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 25))
        self.screen.blit(stage_surf, stage_rect)

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg_text = "YOU WIN!" if self.win else "TIME UP!"
            color = self.COLOR_ORB_GREEN if self.win else self.COLOR_ORB_RED
            msg_surf = self.font_msg.render(msg_text, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _grid_to_screen(self, grid_pos):
        """Converts grid [row, col] to screen [x, y] coordinates."""
        r, c = grid_pos
        x = self.GRID_OFFSET_X + c * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + r * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2
        return [x, y]
        
    def _lerp(self, a, b, t):
        """Linear interpolation."""
        return a + (b - a) * t

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Grid Ninja")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Map pygame keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    movement_action = 0 # No-op
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    movement_action = key_to_action[event.key]
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                    
        # Construct the action for the step function
        # For manual play, we only register the action on the frame it's pressed
        action = [movement_action, 0, 0] # Space and Shift are not used
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # After taking the action, reset the movement to no-op for the next frame
        # This makes it so you have to tap the key for each jump
        movement_action = 0 
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before auto-resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        # Cap the frame rate
        env.clock.tick(env.FPS)
        
    env.close()