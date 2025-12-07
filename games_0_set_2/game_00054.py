
# Generated: 2025-08-27T12:28:08.335558
# Source Brief: brief_00054.md
# Brief Index: 54

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the catcher. Press 'Space' to catch a falling block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling blocks to score points. Match consecutive colors for a bonus, but be careful! Mismatches cost you points, and missing blocks costs you a life."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (15, 23, 42)
    COLOR_GRID = (30, 41, 59)
    COLOR_UI_TEXT = (226, 232, 240)
    COLOR_CATCHER_NEUTRAL = (203, 213, 225)
    BLOCK_COLORS = {
        "red": (239, 68, 68),
        "green": (34, 197, 94),
        "blue": (59, 130, 246),
        "yellow": (234, 179, 8),
    }

    # Game Parameters
    CATCHER_WIDTH = 100
    CATCHER_HEIGHT = 15
    CATCHER_SPEED = 10
    BLOCK_SIZE = 20
    INITIAL_BLOCK_SPEED = 2.0
    MAX_LIVES = 3
    WIN_SCORE = 500
    MAX_STEPS = 10000
    PARTICLE_LIFESPAN = 30
    PARTICLE_COUNT = 20

    # Reward Structure
    REWARD_MOVE = -0.01  # Small penalty for movement to encourage efficiency
    REWARD_CATCH = 1.0
    REWARD_MATCH = 10.0
    REWARD_MISMATCH = -5.0
    REWARD_MISS = -10.0 # Penalty for a block hitting the bottom
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_pop = pygame.font.SysFont("sans-serif", 18, bold=True)

        # State variables are initialized in reset()
        self.catcher_pos = None
        self.catcher_color = None
        self.held_color_name = None
        self.blocks = None
        self.particles = None
        self.pop_texts = None
        self.score = None
        self.lives = None
        self.steps = None
        self.block_speed = None
        self.block_spawn_timer = None
        self.space_pressed_last_frame = None
        self.game_over = None

        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.catcher_pos = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.CATCHER_WIDTH // 2,
            self.SCREEN_HEIGHT - self.CATCHER_HEIGHT - 10,
            self.CATCHER_WIDTH,
            self.CATCHER_HEIGHT
        )
        self.catcher_color = self.COLOR_CATCHER_NEUTRAL
        self.held_color_name = None

        self.blocks = []
        self.particles = []
        self.pop_texts = []
        
        self.score = 0
        self.lives = self.MAX_LIVES
        self.steps = 0
        self.block_speed = self.INITIAL_BLOCK_SPEED
        self.block_spawn_timer = 0
        self.space_pressed_last_frame = False
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # --- Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 3: # Left
            self.catcher_pos.x -= self.CATCHER_SPEED
            reward += self.REWARD_MOVE
        elif movement == 4: # Right
            self.catcher_pos.x += self.CATCHER_SPEED
            reward += self.REWARD_MOVE
        
        self.catcher_pos.x = np.clip(self.catcher_pos.x, 0, self.SCREEN_WIDTH - self.CATCHER_WIDTH)

        # Detect rising edge of space press for catch action
        catch_attempted = space_held and not self.space_pressed_last_frame
        self.space_pressed_last_frame = space_held
        
        if catch_attempted:
            reward += self._handle_catch()

        # --- Update Game State ---
        self._update_blocks()
        self._update_particles()
        self._update_pop_texts()

        # Check for missed blocks
        missed_blocks = [b for b in self.blocks if b['rect'].top > self.SCREEN_HEIGHT]
        for block in missed_blocks:
            self.lives -= 1
            reward += self.REWARD_MISS
            self._create_pop_text("MISS!", block['rect'].center, (255, 100, 100))
            self.blocks.remove(block)
            # sfx: miss_sound

        # Spawn new blocks
        self.block_spawn_timer -= 1
        if self.block_spawn_timer <= 0:
            self._spawn_block()
            self.block_spawn_timer = self.np_random.integers(30, 60)

        # Difficulty scaling
        if self.steps > 0 and self.steps % 100 == 0:
            self.block_speed += 0.05
            
        # --- Termination Check ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += self.REWARD_WIN
            terminated = True
            self.game_over = True
            self._create_pop_text("YOU WIN!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), (255, 255, 100), 60)
        elif self.lives <= 0:
            reward += self.REWARD_LOSS
            terminated = True
            self.game_over = True
            self._create_pop_text("GAME OVER", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), (255, 100, 100), 60)
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_catch(self):
        # Find blocks colliding with the catcher
        catchable_blocks = [b for b in self.blocks if self.catcher_pos.colliderect(b['rect'])]
        
        if not catchable_blocks:
            # sfx: catch_miss_sound
            return 0

        # Catch the lowest block
        block_to_catch = min(catchable_blocks, key=lambda b: b['rect'].bottom)
        
        reward = self.REWARD_CATCH
        
        if self.held_color_name is None or block_to_catch['name'] == self.held_color_name:
            # First catch or a match
            reward += self.REWARD_MATCH
            self.score += 10
            self._create_particles(block_to_catch['rect'].center, block_to_catch['color'])
            self._create_pop_text("+10", block_to_catch['rect'].center, (200, 255, 200))
            # sfx: match_sound
        else:
            # Mismatch
            reward += self.REWARD_MISMATCH
            self.score = max(0, self.score - 5)
            self._create_pop_text("-5", block_to_catch['rect'].center, (255, 150, 150))
            # sfx: mismatch_sound
        
        self.catcher_color = block_to_catch['color']
        self.held_color_name = block_to_catch['name']
        self.blocks.remove(block_to_catch)
        
        return reward

    def _spawn_block(self):
        name, color = random.choice(list(self.BLOCK_COLORS.items()))
        start_x = self.np_random.integers(0, self.SCREEN_WIDTH - self.BLOCK_SIZE)
        rect = pygame.Rect(start_x, -self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
        self.blocks.append({'rect': rect, 'color': color, 'name': name})

    def _update_blocks(self):
        for block in self.blocks:
            block['rect'].y += self.block_speed

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _update_pop_texts(self):
        for t in self.pop_texts[:]:
            t['pos'][1] -= 0.5
            t['lifespan'] -= 1
            if t['lifespan'] <= 0:
                self.pop_texts.remove(t)

    def _create_particles(self, pos, color):
        for _ in range(self.PARTICLE_COUNT):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'color': color,
                'lifespan': self.PARTICLE_LIFESPAN,
                'radius': self.np_random.uniform(2, 5)
            })

    def _create_pop_text(self, text, pos, color, lifespan=30):
        self.pop_texts.append({
            'text': text,
            'pos': list(pos),
            'color': color,
            'lifespan': lifespan
        })

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # --- Game Elements ---
        self._render_particles()
        self._render_blocks()
        self._render_catcher()
        self._render_pop_texts()
        
        # --- UI Overlay ---
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_catcher(self):
        pygame.draw.rect(self.screen, self.catcher_color, self.catcher_pos, border_radius=3)
        # Add a subtle inner border for depth
        inner_rect = self.catcher_pos.inflate(-4, -4)
        pygame.draw.rect(self.screen, tuple(min(255, c + 20) for c in self.catcher_color), inner_rect, border_radius=3)

    def _render_blocks(self):
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / self.PARTICLE_LIFESPAN))
            if alpha > 0:
                color = p['color']
                # Use gfxdraw for anti-aliased circles
                pos_int = (int(p['pos'][0]), int(p['pos'][1]))
                radius_int = int(p['radius'] * (p['lifespan'] / self.PARTICLE_LIFESPAN))
                if radius_int > 0:
                    pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius_int, color)
                    pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, color)


    def _render_pop_texts(self):
        for t in self.pop_texts:
            alpha = int(255 * (t['lifespan'] / 30 if t['lifespan'] < 30 else 1))
            if alpha > 0:
                text_surf = self.font_pop.render(t['text'], True, t['color'])
                text_surf.set_alpha(alpha)
                text_rect = text_surf.get_rect(center=(int(t['pos'][0]), int(t['pos'][1])))
                self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score Display
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives Display
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        lives_rect = lives_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(lives_text, lives_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "block_speed": self.block_speed,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy' or 'windows' depending on your system

    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Block Catcher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op

    while not done:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action = [0, 0, 0] # 0: none, 1: up, 2: down, 3: left, 4: right
        
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        else:
            action[1] = 0
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
        else:
            action[2] = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render for Human ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for human play

    print(f"Game Over! Final Info: {info}")
    env.close()