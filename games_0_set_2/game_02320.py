
# Generated: 2025-08-27T20:01:29.937411
# Source Brief: brief_02320.md
# Brief Index: 2320

        
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


# Barrel class to hold state for each barrel
class Barrel:
    """A class to represent a single barrel with its physics properties."""
    def __init__(self, x, y, radius, barrel_id, np_random):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.radius = radius
        self.id = barrel_id
        self.on_ground = False
        self.angle = np_random.uniform(0, 360) # For visual rotation

    def apply_force(self, force):
        """Applies a force vector to the barrel's velocity."""
        self.vel += force

    def update(self, gravity, friction, ground_y, screen_width):
        """Updates the barrel's state for one physics tick."""
        # Apply gravity
        self.vel.y += gravity
        
        # Update position
        self.pos += self.vel
        
        # Apply friction only when on ground
        if self.on_ground:
            self.vel.x *= (1 - friction)
            if abs(self.vel.x) < 0.1:
                self.vel.x = 0
        
        # Visual rotation based on horizontal velocity
        self.angle += self.vel.x
        
        # Collision with ground
        self.on_ground = False
        if self.pos.y + self.radius > ground_y:
            self.pos.y = ground_y - self.radius
            self.vel.y = 0
            self.on_ground = True
            
        # Collision with walls
        if self.pos.x - self.radius < 0:
            self.pos.x = self.radius
            self.vel.x *= -0.5
        elif self.pos.x + self.radius > screen_width:
            self.pos.x = screen_width - self.radius
            self.vel.x *= -0.5

    def get_rect(self):
        """Returns a pygame.Rect representing the barrel's hitbox."""
        return pygame.Rect(
            self.pos.x - self.radius,
            self.pos.y - self.radius,
            self.radius * 2,
            self.radius * 2
        )

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys apply force to the nearest barrel in that direction. Space also makes the barrel jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide the barrels to their matching green goal zones. Avoid the red pits! The barrels change color from blue (far) to yellow (close) as they approach their goal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    TIME_LIMIT_SECONDS = 60
    
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (30, 40, 50)
    COLOR_GROUND = (52, 63, 75)
    COLOR_PIT = (200, 50, 50)
    COLOR_GOAL = (50, 200, 50)
    COLOR_GOAL_ACTIVE = (150, 255, 150)
    COLOR_TEXT = (230, 230, 230)
    
    COLOR_BARREL_FAR = (60, 120, 220)
    COLOR_BARREL_NEAR = (220, 220, 60)
    COLOR_FOCUS_OUTLINE = (255, 255, 255)

    GRAVITY = 0.4
    FRICTION = 0.05
    PUSH_FORCE = 0.8
    DOWN_FORCE = 0.2
    JUMP_FORCE = -9.0
    
    NUM_BARRELS = 3
    BARREL_RADIUS = 14
    GROUND_Y = 360

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # UI Fonts
        self.font_ui = pygame.font.Font(None, 24)
        self.font_timer = pygame.font.Font(None, 36)
        
        # Game State
        self.barrels = []
        self.pits = []
        self.targets = []
        self.trails = []
        self.max_dist = math.hypot(self.WIDTH, self.HEIGHT)
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_steps = self.TIME_LIMIT_SECONDS * self.FPS
        
        self._setup_level()
        
        self.last_distances = {b.id: b.pos.distance_to(self.targets[b.id].center) for b in self.barrels}
        self.barrels_in_goal = set()
        self.focused_barrel_id = -1
        self.trails = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Creates the static layout of pits, targets, and barrels for an episode."""
        self.barrels.clear()
        self.targets.clear()
        self.pits.clear()

        # Define pits
        self.pits.append(pygame.Rect(280, self.GROUND_Y, 120, self.HEIGHT - self.GROUND_Y))

        # Define targets and barrels
        for i in range(self.NUM_BARRELS):
            # Target zones on the right side
            target_x = self.WIDTH - 120 + (i * 40)
            target_w = self.BARREL_RADIUS * 2 + 4
            self.targets.append(pygame.Rect(target_x, self.GROUND_Y - target_w, target_w, target_w))
            
            # Barrel starting positions on the left side, with some randomness
            start_x = 50 + self.np_random.integers(0, 50) + (i * 10)
            start_y = 150 + self.np_random.integers(0, 50) + (i * 20)
            self.barrels.append(Barrel(start_x, start_y, self.BARREL_RADIUS, i, self.np_random))

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = self._calculate_reward(movement, space_held)
        
        self.steps += 1
        self.score += reward
        terminated = self._check_termination()

        if terminated and not self.game_over: # Win or Timeout
            if len(self.barrels_in_goal) == self.NUM_BARRELS and self.NUM_BARRELS > 0:
                reward += 100 # Win bonus
                # Sound: Win sfx
            else:
                reward -= 10 # Timeout penalty
                self.score -=10
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_reward(self, movement, space_held):
        """Handles game logic updates and calculates the reward for the step."""
        step_reward = -0.01  # Time penalty
        self.focused_barrel_id = -1

        target_barrel = self._get_target_barrel(movement)

        if target_barrel:
            self.focused_barrel_id = target_barrel.id
            # Apply forces based on action
            if movement == 1 or space_held: # Jump
                if target_barrel.on_ground:
                    target_barrel.apply_force(pygame.Vector2(0, self.JUMP_FORCE))
                    # Sound: Jump sfx
            elif movement == 2: # Down
                target_barrel.apply_force(pygame.Vector2(0, self.DOWN_FORCE))
            elif movement == 3: # Left
                target_barrel.apply_force(pygame.Vector2(-self.PUSH_FORCE, 0))
            elif movement == 4: # Right
                target_barrel.apply_force(pygame.Vector2(self.PUSH_FORCE, 0))

        # Update physics and calculate rewards for all barrels
        barrels_to_remove = []
        for barrel in self.barrels:
            barrel.update(self.GRAVITY, self.FRICTION, self.GROUND_Y, self.WIDTH)
            self.trails.append({'pos': barrel.pos.copy(), 'life': 20, 'radius': barrel.radius / 2})

            # Check for falling into a pit
            if barrel.get_rect().collidelist(self.pits) != -1:
                self.game_over = True
                step_reward -= 50
                barrels_to_remove.append(barrel)
                # Sound: Splash/destroy sfx
                continue

            # Calculate distance-based reward
            dist = barrel.pos.distance_to(self.targets[barrel.id].center)
            dist_change = self.last_distances[barrel.id] - dist
            step_reward += 0.1 * (dist_change / self.BARREL_RADIUS) # Normalize by radius
            self.last_distances[barrel.id] = dist

            # Check if in target zone
            if self.targets[barrel.id].contains(barrel.get_rect()):
                if barrel.id not in self.barrels_in_goal:
                    step_reward += 10
                    self.barrels_in_goal.add(barrel.id)
                    # Sound: Goal sfx
            elif barrel.id in self.barrels_in_goal:
                self.barrels_in_goal.remove(barrel.id)
        
        # Remove barrels that fell into pits
        if barrels_to_remove:
            self.barrels = [b for b in self.barrels if b not in barrels_to_remove]
        
        # Update trails
        self.trails = [t for t in self.trails if t['life'] > 0]
        for t in self.trails:
            t['life'] -= 1
        
        return step_reward
    
    def _get_target_barrel(self, movement):
        """Determines which barrel to affect based on the movement action."""
        if not self.barrels or movement == 0:
            return None
        
        try:
            if movement == 1: # Up
                return min(self.barrels, key=lambda b: b.pos.y)
            if movement == 2: # Down
                return max(self.barrels, key=lambda b: b.pos.y)
            if movement == 3: # Left
                return min(self.barrels, key=lambda b: b.pos.x)
            if movement == 4: # Right
                return max(self.barrels, key=lambda b: b.pos.x)
        except (ValueError, IndexError): # Handles case where self.barrels is empty mid-step
            return None
        return None

    def _check_termination(self):
        """Checks for win, loss, or timeout conditions."""
        if self.game_over:
            return True
        if self.NUM_BARRELS > 0 and len(self.barrels_in_goal) == self.NUM_BARRELS:
            return True
        if not self.barrels and self.NUM_BARRELS > 0: # All barrels destroyed
            return True
        if self.steps >= self.max_steps:
            return True
        return False

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
        """Renders all non-UI game elements to the screen."""
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        
        # Draw pits
        for pit in self.pits:
            pygame.draw.rect(self.screen, self.COLOR_PIT, pit)
            
        # Draw targets
        for i, target in enumerate(self.targets):
            color = self.COLOR_GOAL_ACTIVE if i in self.barrels_in_goal else self.COLOR_GOAL
            pygame.gfxdraw.box(self.screen, target, (*color, 100))
            pygame.gfxdraw.rectangle(self.screen, target, color)

        # Draw trails
        for trail in self.trails:
            alpha = int(150 * (trail['life'] / 20))
            color = (*self.COLOR_FOCUS_OUTLINE, alpha)
            radius = int(trail['radius'] * (trail['life'] / 20))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(trail['pos'].x), int(trail['pos'].y), radius, color)

        # Draw barrels
        for barrel in self.barrels:
            dist = barrel.pos.distance_to(self.targets[barrel.id].center)
            progress = np.clip(1 - (dist / (self.max_dist / 2.5)), 0, 1)
            color = pygame.Color(self.COLOR_BARREL_FAR).lerp(self.COLOR_BARREL_NEAR, progress)
            
            pos_int = (int(barrel.pos.x), int(barrel.pos.y))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], barrel.radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], barrel.radius, color)
            
            end_x = pos_int[0] + barrel.radius * math.cos(math.radians(barrel.angle))
            end_y = pos_int[1] + barrel.radius * math.sin(math.radians(barrel.angle))
            pygame.draw.aaline(self.screen, self.COLOR_BG, pos_int, (end_x, end_y), 2)

            if barrel.id == self.focused_barrel_id:
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], barrel.radius + 2, self.COLOR_FOCUS_OUTLINE)

    def _render_ui(self):
        """Renders the UI elements like score and timer."""
        goal_text = f"GOALS: {len(self.barrels_in_goal)} / {self.NUM_BARRELS}"
        text_surf = self.font_ui.render(goal_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        score_text = f"SCORE: {int(self.score)}"
        text_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(centerx=self.WIDTH / 2, y=10)
        self.screen.blit(text_surf, text_rect)

        time_left = (self.max_steps - self.steps) / self.FPS
        time_text = f"{max(0, time_left):.1f}"
        text_surf = self.font_timer.render(time_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(right=self.WIDTH - 10, y=10)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "barrels_in_goal": len(self.barrels_in_goal),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Barrel Roller")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    terminated = False
    total_reward = 0.0

    while not terminated:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0 # unused

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']:.2f}")
    env.close()