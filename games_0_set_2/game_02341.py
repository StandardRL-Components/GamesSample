
# Generated: 2025-08-28T04:29:28.900746
# Source Brief: brief_02341.md
# Brief Index: 2341

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to select a bounce pad. Space to activate the selected pad."
    )

    # Short, user-facing description of the game
    game_description = (
        "A top-down physics puzzle. Maneuver colored boxes onto their matching targets using a limited number of directional bounce pad activations."
    )

    # The game state is static until an action is received.
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    COLOR_BG = (25, 28, 36)
    COLOR_WALL = (80, 80, 90)
    COLOR_PAD = (60, 120, 220)
    COLOR_PAD_SELECTED = (120, 180, 255)
    COLOR_PAD_ARROW = (220, 220, 240)
    COLOR_TEXT = (240, 240, 240)
    
    BOX_COLORS = [(220, 80, 80), (80, 220, 80), (80, 80, 220)]
    TARGET_COLORS = [(70, 30, 30), (30, 70, 30), (30, 30, 70)]

    MAX_BOUNCES = 8
    MAX_STEPS = 1000
    BOX_SIZE = 30
    BOX_SPEED = 10
    WALL_THICKNESS = 10
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 60)
        
        # Game entities setup
        self._define_level()
        
        # Initialize state variables
        self.boxes = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.bounces_left = 0
        self.selected_pad_idx = -1
        self.is_animating = False
        self.np_random = None
        
        # Initialize state by calling reset
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def _define_level(self):
        """Defines the static layout of the level: walls, pads, targets."""
        self.walls = [
            pygame.Rect(0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS),
            pygame.Rect(0, self.SCREEN_HEIGHT - self.WALL_THICKNESS, self.SCREEN_WIDTH, self.WALL_THICKNESS),
            pygame.Rect(0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT),
            pygame.Rect(self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT),
            pygame.Rect(200, 150, self.WALL_THICKNESS, 100),
            pygame.Rect(430, 150, self.WALL_THICKNESS, 100),
        ]

        pad_size = 40
        self.bounce_pads = [
            {'pos': (self.SCREEN_WIDTH / 2, 50), 'dir': pygame.Vector2(0, 1)},  # Down
            {'pos': (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50), 'dir': pygame.Vector2(0, -1)}, # Up
            {'pos': (60, self.SCREEN_HEIGHT / 2), 'dir': pygame.Vector2(1, 0)},  # Right
            {'pos': (self.SCREEN_WIDTH - 60, self.SCREEN_HEIGHT / 2), 'dir': pygame.Vector2(-1, 0)}, # Left
        ]
        
        self.targets = [
            {'pos': (120, 100), 'color': self.TARGET_COLORS[0]},
            {'pos': (self.SCREEN_WIDTH - 120 - self.BOX_SIZE, 100), 'color': self.TARGET_COLORS[1]},
            {'pos': (self.SCREEN_WIDTH / 2 - self.BOX_SIZE / 2, 200), 'color': self.TARGET_COLORS[2]},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.bounces_left = self.MAX_BOUNCES
        self.selected_pad_idx = -1
        self.is_animating = False
        self.particles.clear()
        
        initial_box_positions = [
            (120, 280),
            (self.SCREEN_WIDTH - 120 - self.BOX_SIZE, 280),
            (300, 100)
        ]
        
        self.boxes = []
        for i in range(len(self.targets)):
            box_pos = initial_box_positions[i]
            box_color = self.BOX_COLORS[i]
            target_pos = self.targets[i]['pos']
            self.boxes.append({
                'rect': pygame.Rect(box_pos[0], box_pos[1], self.BOX_SIZE, self.BOX_SIZE),
                'vel': pygame.Vector2(0, 0),
                'color': box_color,
                'target_rect': pygame.Rect(target_pos[0], target_pos[1], self.BOX_SIZE, self.BOX_SIZE),
                'on_target_last_frame': False
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.is_animating:
            self._update_physics_frame()
        else:
            reward = self._handle_player_action(action)

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # End due to step limit
            reward -= 50 # Penalty for running out of time
            self.score -= 50
            self.game_over = True
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _handle_player_action(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # Action 1-4 (arrows) select a bounce pad
        if 1 <= movement <= 4:
            self.selected_pad_idx = movement - 1

        # Space activates the selected pad
        if space_held and self.selected_pad_idx != -1 and self.bounces_left > 0:
            self.bounces_left -= 1
            reward = -0.1 # Cost for using a bounce
            
            # Start animation
            self.is_animating = True
            pad = self.bounce_pads[self.selected_pad_idx]
            for box in self.boxes:
                box['vel'] = pad['dir'] * self.BOX_SPEED
                box['on_target_last_frame'] = box['rect'].colliderect(box['target_rect'])

            # Visual effect for pad activation
            # sfx: Pad activation sound
            self._create_particles(pad['pos'], self.COLOR_PAD_SELECTED, 20)
            
        return reward

    def _update_physics_frame(self):
        active_boxes = 0
        for box in self.boxes:
            if box['vel'].length() > 0:
                active_boxes += 1
                
                # Move box
                box['rect'].x += box['vel'].x
                box['rect'].y += box['vel'].y

                # Wall collision
                for wall in self.walls:
                    if box['rect'].colliderect(wall):
                        # sfx: Box hit wall sound
                        box['vel'] = pygame.Vector2(0, 0)
                        # Clamp position to be outside wall
                        if wall.width > wall.height: # Horizontal wall
                            if box['rect'].centery < self.SCREEN_HEIGHT / 2: box['rect'].top = wall.bottom
                            else: box['rect'].bottom = wall.top
                        else: # Vertical wall
                            if box['rect'].centerx < self.SCREEN_WIDTH / 2: box['rect'].left = wall.right
                            else: box['rect'].right = wall.left
                        break
                
                # Box-Box collision
                for other_box in self.boxes:
                    if box != other_box and box['rect'].colliderect(other_box['rect']):
                        # sfx: Box hit box sound
                        box['vel'] = pygame.Vector2(0, 0)
                        other_box['vel'] = pygame.Vector2(0, 0)
                        # A simple stop-on-contact model for puzzle predictability
                        break

        if active_boxes == 0:
            self.is_animating = False
            self._check_post_animation_state()

    def _check_post_animation_state(self):
        all_on_target = True
        reward = 0
        
        for box in self.boxes:
            is_on_target = box['rect'].colliderect(box['target_rect'])
            if not is_on_target:
                all_on_target = False
            # Reward for a box newly arriving on target
            if is_on_target and not box['on_target_last_frame']:
                # sfx: Box on target success sound
                reward += 5
                self._create_particles(box['rect'].center, box['color'], 15, life=30)

        self.score += reward

        if all_on_target:
            # sfx: Level complete fanfare
            self.score += 50
            self.game_over = True
            self._create_particles((self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), (255,215,0), 100, life=90)
        elif self.bounces_left <= 0:
            # sfx: Failure sound
            self.score -= 50
            self.game_over = True

    def _create_particles(self, pos, color, count, life=20):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': life,
                'max_life': life,
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Draw targets
        for target in self.targets:
            pygame.draw.rect(self.screen, target['color'], (target['pos'][0], target['pos'][1], self.BOX_SIZE, self.BOX_SIZE))
        
        # Draw bounce pads
        for i, pad in enumerate(self.bounce_pads):
            color = self.COLOR_PAD_SELECTED if i == self.selected_pad_idx else self.COLOR_PAD
            self._draw_bounce_pad(pad['pos'], pad['dir'], color)
        
        # Draw particles
        self._update_particles()
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(5 * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'].x), int(p['pos'].y), size, (*p['color'], alpha)
                )

        # Draw boxes
        for box in self.boxes:
            pygame.draw.rect(self.screen, box['color'], box['rect'])
            pygame.draw.rect(self.screen, tuple(min(c+50, 255) for c in box['color']), box['rect'], 2)

    def _draw_bounce_pad(self, pos, direction, color):
        size = 20
        p1 = pygame.Vector2(pos)
        p2 = p1 + direction.rotate(140) * size
        p3 = p1 + direction.rotate(-140) * size
        
        points = [
            (int(p1.x), int(p1.y)),
            (int(p2.x), int(p2.y)),
            (int(p3.x), int(p3.y))
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_ui(self):
        # Bounces Left
        text_surf = self.font_ui.render(f"Bounces Left: {self.bounces_left}", True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (20, 20))
        
        # Score
        score_surf = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 20, 20))
        
        # Game Over Messages
        if self.game_over:
            all_on_target = all(b['rect'].colliderect(b['target_rect']) for b in self.boxes)
            if all_on_target:
                msg = "SUCCESS!"
                color = (100, 255, 100)
            else:
                msg = "OUT OF BOUNCES"
                color = (255, 100, 100)
                
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_surf = pygame.Surface(msg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))
            self.screen.blit(bg_surf, msg_rect.topleft)
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bounces_left": self.bounces_left,
            "is_animating": self.is_animating,
            "selected_pad": self.selected_pad_idx,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("BounceBox Gym Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                if done: continue
                
                if event.key == pygame.K_UP:
                    action[0] = 2 # Selects pad index 1 (UP)
                elif event.key == pygame.K_DOWN:
                    action[0] = 1 # Selects pad index 0 (DOWN)
                elif event.key == pygame.K_LEFT:
                    action[0] = 4 # Selects pad index 3 (LEFT)
                elif event.key == pygame.K_RIGHT:
                    action[0] = 3 # Selects pad index 2 (RIGHT)
                elif event.key == pygame.K_SPACE:
                    action[1] = 1 # Activate

        # If the env is animating, we need to keep stepping with no-op to see it
        if info.get('is_animating', False):
            action = [0, 0, 0]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        clock.tick(30) # Run at 30 FPS

    env.close()