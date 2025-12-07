# Generated: 2025-08-27T23:26:42.505593
# Source Brief: brief_03464.md
# Brief Index: 3464

        
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
    A Gymnasium environment for a fruit-slicing arcade game.

    The agent controls a vertical slicer that moves horizontally across the screen.
    The goal is to slice falling fruits to score points while avoiding bombs.
    Slicing a fruit awards points, while slicing a bomb results in losing a life.
    The game ends when the player reaches the target score, loses all lives, or the
    time limit (max steps) is reached.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ←→ to move the slicer horizontally. Press space to slice."
    )

    game_description = (
        "Slice falling fruit to score points, avoid bombs, and reach a target score "
        "before running out of lives in this fast-paced, grid-based arcade game."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.GRID_CELL_WIDTH = self.WIDTH // self.GRID_COLS
        self.GRID_CELL_HEIGHT = self.HEIGHT // self.GRID_ROWS
        
        self.MAX_LIVES = 3
        self.WIN_SCORE = 500
        self.MAX_STEPS = 1000
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_SLICER = (255, 255, 255)
        self.COLOR_SLICE_EFFECT = (175, 238, 238)
        self.COLOR_BOMB = (30, 30, 30)
        self.COLOR_FUSE = (255, 100, 0)
        self.COLOR_SKULL = (220, 220, 220)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.FRUIT_PALETTE = {
            "apple": (220, 50, 50),
            "orange": (255, 165, 0),
            "banana": (255, 225, 53),
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.slicer_col = 0
        self.objects = []
        self.particles = []
        self.previous_space_state = False
        self.slice_effect_timer = 0
        self.spawn_timer = 0
        self.base_speed = 0
        self.speed_increase = 0
        self.score_milestones_cleared = 0
        
        # Initialize state
        # self.reset() is called by the test runner, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        
        self.slicer_col = self.GRID_COLS // 2
        self.objects = []
        self.particles = []
        
        self.previous_space_state = False
        self.slice_effect_timer = 0
        self.spawn_timer = 0
        self.base_speed = 1.5
        self.speed_increase = 0.0
        self.score_milestones_cleared = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.auto_advance:
            self.clock.tick(30)
            
        reward = 0
        
        # 1. Handle player input
        self._handle_input(action)

        # 2. Perform slice action on space press (rising edge)
        space_held = action[1] == 1
        slice_triggered = space_held and not self.previous_space_state
        if slice_triggered:
            reward += self._perform_slice()
        self.previous_space_state = space_held
        
        # 3. Update game state
        self._update_game_state()
        
        # 4. Calculate milestone rewards
        old_milestones = self.score_milestones_cleared
        new_milestones = self.score // 100
        if new_milestones > old_milestones:
            reward += 10 * (new_milestones - old_milestones)
            self.score_milestones_cleared = new_milestones

        # 5. Check for termination
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.lives <= 0:
            reward += -100
            terminated = True
            self.game_over = True
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Per Gymnasium API, truncated episodes are also terminated
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.slicer_col = max(0, self.slicer_col - 1)
        elif movement == 4:  # Right
            self.slicer_col = min(self.GRID_COLS - 1, self.slicer_col + 1)

    def _perform_slice(self):
        # SFX: Slice sound
        self.slice_effect_timer = 5  # frames
        slicer_x_center = int((self.slicer_col + 0.5) * self.GRID_CELL_WIDTH)
        slice_reward = 0
        
        sliced_indices = []
        for i, obj in enumerate(self.objects):
            obj_x_pos = obj['pos'].x
            if abs(obj_x_pos - slicer_x_center) < obj['radius']:
                sliced_indices.append(i)

        for i in sorted(sliced_indices, reverse=True):
            obj = self.objects.pop(i)
            if obj['type'] == 'fruit':
                # SFX: Juicy slice sound
                slice_reward += 1
                self.score += 10
                self._create_particles(obj['pos'], obj['color'], 30, 'fruit')
            elif obj['type'] == 'bomb':
                # SFX: Explosion sound
                slice_reward -= 5
                self.lives -= 1
                self._create_particles(obj['pos'], self.COLOR_FUSE, 50, 'bomb')
        
        return slice_reward

    def _update_game_state(self):
        # Update difficulty
        self.speed_increase = (self.steps // 100) * 0.2

        # Update slice effect timer
        self.slice_effect_timer = max(0, self.slice_effect_timer - 1)
        
        # Move and manage objects
        for obj in self.objects[:]:
            obj['pos'].y += self.base_speed + self.speed_increase
            if obj['pos'].y - obj['radius'] > self.HEIGHT:
                self.objects.remove(obj)
        
        # Spawn new objects
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self._spawn_object()
            self.spawn_timer = self.np_random.integers(20, 40)

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'].y += 0.1  # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _spawn_object(self):
        col = self.np_random.integers(0, self.GRID_COLS)
        x_pos = (col + 0.5) * self.GRID_CELL_WIDTH
        
        obj = {
            'pos': pygame.Vector2(x_pos, -20),
            'radius': self.np_random.integers(15, 25)
        }
        
        if self.np_random.random() < 0.2: # 20% chance of bomb
            obj.update({
                'type': 'bomb',
                'color': self.COLOR_BOMB,
            })
        else:
            fruit_type = self.np_random.choice(list(self.FRUIT_PALETTE.keys()))
            obj.update({
                'type': 'fruit',
                'fruit_type': fruit_type,
                'color': self.FRUIT_PALETTE[fruit_type],
            })
        self.objects.append(obj)

    def _create_particles(self, pos, color, count, p_type):
        for _ in range(count):
            if p_type == 'fruit':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
                radius = self.np_random.integers(2, 5)
            else: # bomb
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 8)
                vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
                radius = self.np_random.integers(3, 7)
                color = random.choice([self.COLOR_FUSE, (255, 255, 0), (150, 150, 150)])

            self.particles.append({
                'pos': pygame.Vector2(pos), # FIX: pygame.Vector2 does not have .copy(), create new instance instead
                'vel': vel,
                'color': color,
                'radius': radius,
                'lifespan': self.np_random.integers(20, 40)
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for i in range(1, self.GRID_COLS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i * self.GRID_CELL_WIDTH, 0), (i * self.GRID_CELL_WIDTH, self.HEIGHT))
        for i in range(1, self.GRID_ROWS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i * self.GRID_CELL_HEIGHT), (self.WIDTH, i * self.GRID_CELL_HEIGHT))
            
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40))
            color_with_alpha = p['color'] + (alpha,)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color_with_alpha)
            except TypeError: # Handle potential color tuple with alpha
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])


        # Draw objects
        for obj in self.objects:
            pos = (int(obj['pos'].x), int(obj['pos'].y))
            if obj['type'] == 'fruit':
                self._draw_fruit(pos, obj['color'], obj['radius'], obj['fruit_type'])
            else:
                self._draw_bomb(pos, obj['radius'])

        # Draw slicer and slice effect
        slicer_x = int((self.slicer_col + 0.5) * self.GRID_CELL_WIDTH)
        pygame.draw.line(self.screen, self.COLOR_SLICER, (slicer_x, 0), (slicer_x, self.HEIGHT), 2)
        if self.slice_effect_timer > 0:
            alpha = int(255 * (self.slice_effect_timer / 5))
            swoosh_surface = pygame.Surface((self.GRID_CELL_WIDTH, self.HEIGHT), pygame.SRCALPHA)
            swoosh_surface.fill((self.COLOR_SLICE_EFFECT[0], self.COLOR_SLICE_EFFECT[1], self.COLOR_SLICE_EFFECT[2], alpha))
            self.screen.blit(swoosh_surface, (slicer_x - self.GRID_CELL_WIDTH // 2, 0))

    def _draw_fruit(self, pos, color, radius, fruit_type):
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        # Add a little stem/leaf for visual flair
        pygame.draw.line(self.screen, (139, 69, 19), (pos[0], pos[1] - radius), (pos[0], pos[1] - radius - 5), 3)

    def _draw_bomb(self, pos, radius):
        # Bomb body
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_BOMB)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_BOMB)
        # Fuse
        fuse_end = (pos[0] + radius, pos[1] - radius)
        pygame.draw.line(self.screen, self.COLOR_FUSE, (pos[0] + 5, pos[1] - radius), fuse_end, 3)
        pygame.gfxdraw.filled_circle(self.screen, fuse_end[0], fuse_end[1], 3, (255, 255, 0)) # Spark
        # Skull icon
        cx, cy = pos[0], pos[1]
        r = radius * 0.5
        pygame.draw.rect(self.screen, self.COLOR_SKULL, (cx - r/2, cy - r/2, r, r))
        pygame.gfxdraw.filled_circle(self.screen, int(cx - r/4), int(cy - r/4), int(r/4), self.COLOR_BOMB)
        pygame.gfxdraw.filled_circle(self.screen, int(cx + r/4), int(cy - r/4), int(r/4), self.COLOR_BOMB)
        
    def _render_ui(self):
        # Score
        self._draw_text(f"Score: {self.score}", (10, 10), self.font_ui)
        
        # Lives
        lives_text = f"Lives: {self.lives}"
        text_width = self.font_ui.size(lives_text)[0]
        self._draw_text(lives_text, (self.WIDTH - text_width - 10, 10), self.font_ui)
        
        # Game Over message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2), self.font_game_over, color, center=True)

    def _draw_text(self, text, pos, font, color=None, center=False):
        if color is None:
            color = self.COLOR_TEXT
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, color)
        
        if center:
            text_rect = text_surf.get_rect(center=pos)
        else:
            text_rect = text_surf.get_rect(topleft=pos)
        
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for self-testing and not part of the standard Gym API
        print("Running implementation validation...")
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

if __name__ == '__main__':
    # This block allows you to run the game directly to test it
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Play ---
    # Use arrow keys to move, space to slice
    # The game will run in a Pygame window
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Slicer")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Action state
    movement = 0 # 0: none, 3: left, 4: right
    space_pressed = 0 # 0: released, 1: held
    
    print(env.user_guide)

    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_pressed = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT and movement == 3:
                    movement = 0
                elif event.key == pygame.K_RIGHT and movement == 4:
                    movement = 0
                elif event.key == pygame.K_SPACE:
                    space_pressed = 0

        action = [movement, space_pressed, 0] # Shift is not used
        obs, reward, terminated, truncated, info = env.step(action)

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if info.get('lives', env.MAX_LIVES) <= 0:
            print(f"Game Over! Final Score: {info['score']}")
            
        if info.get('score', 0) >= env.WIN_SCORE:
            print(f"You Win! Final Score: {info['score']}")

        clock.tick(30) # Limit human play to 30 FPS

    print("Closing game.")
    env.close()