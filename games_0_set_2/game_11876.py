import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:39:31.421069
# Source Brief: brief_01876.md
# Brief Index: 1876
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    GameEnv: Twin Platforms
    Control two platforms to catch falling geometric objects.
    Synchronizing their movement direction on the same frame activates a temporary score multiplier,
    visualized by a glowing beam connecting the platforms.
    The goal is to reach a score of 50 before the 1000-step time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control two platforms to catch falling objects. "
        "Move both platforms in the same direction to activate a temporary score multiplier."
    )
    user_guide = (
        "Use ←/→ for the left platform. Use 'space' to move the right platform left and 'shift' to move it right."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000
    WIN_SCORE = 50

    # Colors
    COLOR_BG = (15, 18, 28)
    COLOR_PLATFORM = (240, 240, 255)
    COLOR_PLATFORM_GLOW = (180, 180, 220, 50)
    COLOR_SYNC_BEAM = (255, 220, 0)
    COLOR_SYNC_GLOW = (255, 220, 0, 60)
    COLOR_SCORE_TEXT = (220, 220, 230)
    OBJECT_COLORS = [
        (255, 80, 120),   # Hot Pink
        (80, 255, 150),   # Mint Green
        (80, 150, 255),   # Sky Blue
        (255, 180, 50),   # Orange
    ]

    # Game element properties
    PLATFORM_WIDTH = 90
    PLATFORM_HEIGHT = 12
    PLATFORM_Y = HEIGHT - 30
    PLATFORM_SPEED = 12
    
    OBJECT_MIN_SIZE = 8
    OBJECT_MAX_SIZE = 12
    
    INITIAL_FALL_SPEED = 2.0
    INITIAL_SPAWN_INTERVAL = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.platform1_x = 0
        self.platform2_x = 0
        
        self.objects = []
        self.particles = []

        self.sync_timer = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.spawn_interval = self.INITIAL_SPAWN_INTERVAL
        self.steps_since_spawn = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.platform1_x = self.WIDTH / 4 - self.PLATFORM_WIDTH / 2
        self.platform2_x = self.WIDTH * 3 / 4 - self.PLATFORM_WIDTH / 2

        self.objects = []
        self.particles = []

        self.sync_timer = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.spawn_interval = self.INITIAL_SPAWN_INTERVAL
        self.steps_since_spawn = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # On subsequent steps after termination/truncation, return the final state
            terminated = self.score >= self.WIN_SCORE
            truncated = self.steps >= self.MAX_STEPS
            return self._get_observation(), 0, terminated, truncated, self._get_info()

        self.steps += 1
        reward = 0

        # --- Game Logic ---
        reward += self._handle_input(action)
        reward += self._update_objects()
        self._update_particles()
        self._update_difficulty()

        if self.sync_timer > 0:
            self.sync_timer -= 1
        
        # --- Termination ---
        terminated = self.score >= self.WIN_SCORE
        truncated = self.steps >= self.MAX_STEPS
        
        if (terminated or truncated) and not self.game_over:
            self.game_over = True
            if terminated:
                reward += 100  # Goal-oriented reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement_cmd = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        p1_move = 0
        p2_move = 0
        reward = 0

        # Left platform control (from movement_cmd)
        if movement_cmd == 3: # Left
            self.platform1_x -= self.PLATFORM_SPEED
            p1_move = -1
        elif movement_cmd == 4: # Right
            self.platform1_x += self.PLATFORM_SPEED
            p1_move = 1

        # Right platform control (from space/shift)
        if shift_held: # Right (priority)
            self.platform2_x += self.PLATFORM_SPEED
            p2_move = 1
        elif space_held: # Left
            self.platform2_x -= self.PLATFORM_SPEED
            p2_move = -1
        
        # Clamp platform positions
        self.platform1_x = max(0, min(self.platform1_x, self.WIDTH - self.PLATFORM_WIDTH))
        self.platform2_x = max(0, min(self.platform2_x, self.WIDTH - self.PLATFORM_WIDTH))
        
        # Sync bonus logic
        if p1_move != 0 and p1_move == p2_move:
            if self.sync_timer == 0:
                reward += 2.0  # Reward for activating sync
            self.sync_timer = 30 # Sync lasts for 1 second (30 frames)

        return reward

    def _update_objects(self):
        reward = 0
        
        # Spawn new objects
        self.steps_since_spawn += 1
        if self.steps_since_spawn >= self.spawn_interval:
            self.steps_since_spawn = 0
            self._spawn_object()

        # Update existing objects
        remaining_objects = []
        for obj in self.objects:
            obj['y'] += self.fall_speed
            
            p1_rect = pygame.Rect(self.platform1_x, self.PLATFORM_Y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
            p2_rect = pygame.Rect(self.platform2_x, self.PLATFORM_Y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
            obj_rect = pygame.Rect(obj['x'] - obj['size'], obj['y'] - obj['size'], obj['size']*2, obj['size']*2)

            caught = False
            if p1_rect.colliderect(obj_rect) or p2_rect.colliderect(obj_rect):
                caught = True
                
                # --- Catch ---
                reward += 0.1
                self.score += 1
                if self.sync_timer > 0:
                    reward += 0.1 # Bonus for sync catch
                    self.score += 1 # Bonus score
                self._create_particles(obj['x'], obj['y'], obj['color'], 30, 4)
                
            elif obj['y'] > self.HEIGHT:
                # --- Drop ---
                reward -= 0.3
                self.score = max(0, self.score - 2)
                self._create_particles(obj['x'], self.HEIGHT - 5, (100, 100, 110), 10, 1)
            else:
                remaining_objects.append(obj)
        
        self.objects = remaining_objects
        return reward

    def _spawn_object(self):
        size = self.np_random.integers(self.OBJECT_MIN_SIZE, self.OBJECT_MAX_SIZE + 1)
        color_index = self.np_random.integers(len(self.OBJECT_COLORS))
        self.objects.append({
            'x': self.np_random.integers(size, self.WIDTH - size),
            'y': -size,
            'size': size,
            'color': self.OBJECT_COLORS[color_index],
            'shape': self.np_random.choice(['circle', 'square'])
        })

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 100 == 0:
            self.fall_speed += 0.05
            self.spawn_interval = max(20, self.spawn_interval - 2)

    def _create_particles(self, x, y, color, count, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'lifetime': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vx'] *= 0.95 # Drag
            p['vy'] *= 0.95
            p['lifetime'] -= 1
            if p['lifetime'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Render sync beam
        if self.sync_timer > 0:
            p1_center = (self.platform1_x + self.PLATFORM_WIDTH / 2, self.PLATFORM_Y + self.PLATFORM_HEIGHT / 2)
            p2_center = (self.platform2_x + self.PLATFORM_WIDTH / 2, self.PLATFORM_Y + self.PLATFORM_HEIGHT / 2)
            
            # Glow effect
            alpha_fade = min(1.0, self.sync_timer / 15.0)
            glow_color = (*self.COLOR_SYNC_GLOW[:3], int(self.COLOR_SYNC_GLOW[3] * alpha_fade))
            pygame.draw.line(self.screen, glow_color, p1_center, p2_center, 15)
            
            # Main beam
            beam_color = (*self.COLOR_SYNC_BEAM, int(255 * alpha_fade))
            pygame.draw.line(self.screen, beam_color, p1_center, p2_center, 3)

        # Render platforms
        p1_rect = (self.platform1_x, self.PLATFORM_Y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
        p2_rect = (self.platform2_x, self.PLATFORM_Y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
        
        # Platform glow
        glow_surf = pygame.Surface((self.PLATFORM_WIDTH + 10, self.PLATFORM_HEIGHT + 10), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLATFORM_GLOW, glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, (self.platform1_x - 5, self.PLATFORM_Y - 5))
        self.screen.blit(glow_surf, (self.platform2_x - 5, self.PLATFORM_Y - 5))
        
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p1_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p2_rect, border_radius=4)
        
        # Render objects
        for obj in self.objects:
            x, y, size = int(obj['x']), int(obj['y']), int(obj['size'])
            if obj['shape'] == 'circle':
                pygame.gfxdraw.aacircle(self.screen, x, y, size, obj['color'])
                pygame.gfxdraw.filled_circle(self.screen, x, y, size, obj['color'])
            else: # square
                rect = pygame.Rect(x - size, y - size, size * 2, size * 2)
                pygame.draw.rect(self.screen, obj['color'], rect)

        # Render particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifetime'] / 40.0))
            color_with_alpha = (*p['color'], int(alpha))
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p['x'] - p['size'], p['y'] - p['size']))

    def _render_ui(self):
        # Render score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_SCORE_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Render steps
        step_text = self.font_small.render(f"STEP: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_SCORE_TEXT)
        self.screen.blit(step_text, (self.WIDTH - step_text.get_width() - 15, 15))
        
        # Render sync bonus text when active
        if self.sync_timer > 0:
            sync_text_str = "SYNC BONUS!"
            if self.score >= 1 and self.sync_timer > 20: # Show only on recent catches
                 sync_text_str = "SYNC CATCH! x2"

            sync_text = self.font_large.render(sync_text_str, True, self.COLOR_SYNC_BEAM)
            pos_x = self.WIDTH / 2 - sync_text.get_width() / 2
            self.screen.blit(sync_text, (pos_x, 50))
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for internal testing and is not part of the standard Gym API
        print("✓ Validating implementation...")
        # Test reset
        obs, info = self.reset()
        assert self.observation_space.contains(obs), "Reset observation is not in the observation space"
        assert isinstance(info, dict), "Reset info is not a dictionary"
        
        # Test action space
        assert self.action_space.shape == (3,), f"Action space shape is {self.action_space.shape}, expected (3,)"
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {self.action_space.nvec.tolist()}, expected [5, 2, 2]"
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert self.observation_space.contains(obs), "Step observation is not in the observation space"
        assert isinstance(reward, (int, float)), "Reward is not a number"
        assert isinstance(term, bool), "Terminated flag is not a boolean"
        assert isinstance(trunc, bool), "Truncated flag is not a boolean"
        assert isinstance(info, dict), "Step info is not a dictionary"
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Note: The controls here are mapped for human convenience and might differ from the agent's action space.
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    obs, info = env.reset()
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Twin Platforms")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Left Platform:  A (Left), D (Right)")
    print("Right Platform: Left Arrow (Left), Right Arrow (Right)")
    print("----------------------\n")

    while running:
        # Agent action mapping for manual play
        movement = 0 # no-op
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        
        # This mapping is different from the agent's but more intuitive for humans.
        # Left Platform Control (maps to agent's movement_cmd)
        if keys[pygame.K_a]:
            movement = 3 # left
        elif keys[pygame.K_d]:
            movement = 4 # right
            
        # Right Platform Control (maps to agent's space/shift)
        if keys[pygame.K_LEFT]:
            space = 1
        if keys[pygame.K_RIGHT]:
            shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            end_reason = "won" if terminated else "time limit"
            print(f"Game Over! You {end_reason}. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)
        
    env.close()