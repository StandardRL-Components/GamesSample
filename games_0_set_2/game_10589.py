import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:39:37.000594
# Source Brief: brief_00589.md
# Brief Index: 589
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Ant Colony Resource Management Game
    
    The player controls a selector to assign 5 worker ants to 3 replenishable
    food resources. The goal is to collect 50 units of food within 30 turns.
    An ant's collection efficiency (momentum) increases with each successful haul.
    
    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Selector Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - action[1]: Select/Assign (Spacebar) (0: released, 1: held)
    - action[2]: Cycle Selected Ant (Shift) (0: released, 1: held)
    
    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Manage an ant colony by assigning workers to gather food from resources. Collect enough food before the turns run out to win."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move the selector. Press space to select an ant or assign it to a resource. Press shift to cycle through your ants."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        # Game Parameters
        self.MAX_TURNS = 30
        self.WIN_SCORE = 50
        self.MAX_STEPS = 1500 # Failsafe termination
        self.N_ANTS = 5
        self.N_RESOURCES = 3
        
        # Colors
        self.COLOR_BG = (20, 40, 30)
        self.COLOR_NEST = (139, 69, 19)
        self.COLOR_RESOURCE = (60, 179, 113)
        self.COLOR_RESOURCE_DEPLETED = (80, 100, 90)
        self.COLOR_ANT = (10, 10, 10)
        self.COLOR_SELECTED_ANT = (255, 69, 0)
        self.COLOR_SELECTOR = (0, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_BAR_BG = (50, 50, 50)
        self.COLOR_BAR_FG = (255, 215, 0)
        self.COLOR_WIN = (173, 255, 47)
        self.COLOR_LOSE = (220, 20, 60)

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)
        self.font_resource = pygame.font.SysFont("Consolas", 14, bold=True)
        
        # --- Entity Positions ---
        self.nest_pos = np.array([self.WIDTH // 2, self.HEIGHT - 50])
        self.resource_pos = [np.array([120 + i * 200, 80]) for i in range(self.N_RESOURCES)]
        self.ant_start_pos = [self.nest_pos + np.array([ (i - 2) * 30, 20]) for i in range(self.N_ANTS)]
        
        self.entity_radius = 12
        self.selector_speed = 8

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.turns = 0
        self.score = 0.0
        self.game_over = False
        self.win_status = None # 'win', 'lose', 'timeout'
        
        self.resource_levels = []
        self.resource_max_level = 20.0
        
        self.ant_momentum = []
        self.ant_positions = []
        self.ant_targets = [] # -1 for nest, 0-2 for resources
        
        self.selected_ant = -1
        self.selector_pos = np.array([0.0, 0.0])
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.particles = []

        # self.reset() is called by the wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.turns = 0
        self.score = 0.0
        self.game_over = False
        self.win_status = None
        
        self.resource_levels = [self.resource_max_level] * self.N_RESOURCES
        
        self.ant_momentum = [1.0] * self.N_ANTS
        self.ant_positions = [pos.astype(float) for pos in self.ant_start_pos]
        self.ant_targets = [-1] * self.N_ANTS
        
        self.selected_ant = -1
        self.selector_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        reward = 0
        
        self._update_selector(movement)
        
        if shift_pressed:
            # Sfx: UI_Cycle.wav
            self.selected_ant = (self.selected_ant + 1) % self.N_ANTS
        
        if space_pressed:
            reward += self._handle_selection_and_assignment()

        self._update_ants()
        self._update_resources()
        self._update_particles()
        
        self.steps += 1
        reward += self._calculate_reward()
        terminated = self._check_termination()
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_selection_and_assignment(self):
        # Check for ant selection
        for i in range(self.N_ANTS):
            if np.linalg.norm(self.selector_pos - self.ant_positions[i]) < self.entity_radius:
                self.selected_ant = i
                # Sfx: UI_Select.wav
                return 0 # No reward for just selecting

        # Check for resource assignment
        if self.selected_ant != -1:
            for i in range(self.N_RESOURCES):
                if np.linalg.norm(self.selector_pos - self.resource_pos[i]) < self.entity_radius + 5:
                    # Sfx: Ant_Assign.wav
                    return self._assign_ant_to_resource(self.selected_ant, i)
        
        # Sfx: UI_Fail.wav
        return 0

    def _assign_ant_to_resource(self, ant_idx, res_idx):
        self.turns += 1
        
        momentum = self.ant_momentum[ant_idx]
        res_level = self.resource_levels[res_idx]
        
        food_collected = momentum * res_level * 0.1
        food_collected = max(0, food_collected)
        
        if food_collected > 0:
            # Sfx: Collect_Food.wav
            self.score += food_collected
            self.resource_levels[res_idx] -= food_collected
            self.ant_momentum[ant_idx] += 0.1
            self._create_particles(self.resource_pos[res_idx], int(food_collected * 2))

        self.ant_targets[ant_idx] = res_idx
        self.selected_ant = -1 # Deselect after assignment
        
        return food_collected * 0.1 # Continuous reward

    def _update_selector(self, movement):
        if movement == 1: self.selector_pos[1] -= self.selector_speed
        elif movement == 2: self.selector_pos[1] += self.selector_speed
        elif movement == 3: self.selector_pos[0] -= self.selector_speed
        elif movement == 4: self.selector_pos[0] += self.selector_speed
        self.selector_pos[0] = np.clip(self.selector_pos[0], 0, self.WIDTH)
        self.selector_pos[1] = np.clip(self.selector_pos[1], 0, self.HEIGHT)

    def _update_ants(self):
        for i in range(self.N_ANTS):
            target_idx = self.ant_targets[i]
            target_pos = self.nest_pos if target_idx == -1 else self.resource_pos[target_idx]
            
            direction = target_pos - self.ant_positions[i]
            dist = np.linalg.norm(direction)
            
            if dist > 1:
                self.ant_positions[i] += direction / dist * 3.0 # Ant speed
            
            # If ant reaches a resource, send it back to the nest
            if target_idx != -1 and dist < 5:
                self.ant_targets[i] = -1

    def _update_resources(self):
        for i in range(self.N_RESOURCES):
            replenish_amount = self.resource_levels[i] * 0.01
            self.resource_levels[i] = min(self.resource_max_level, self.resource_levels[i] + replenish_amount)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[3] > 0]
        for p in self.particles:
            p[0] += p[2][0] # Update x
            p[1] += p[2][1] # Update y
            p[3] -= 1       # Decrease lifetime

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(20, 40)
            self.particles.append([pos[0], pos[1], velocity, lifetime])

    def _calculate_reward(self):
        # This is for terminal rewards
        if self.score >= self.WIN_SCORE and self.win_status is None:
            self.win_status = 'win'
            # Sfx: Game_Win.wav
            return 100
        if self.turns >= self.MAX_TURNS and self.win_status is None:
            self.win_status = 'lose'
            # Sfx: Game_Lose.wav
            return -100
        if self.steps >= self.MAX_STEPS and self.win_status is None:
            self.win_status = 'timeout'
            return -100 # Also a loss condition
        return 0

    def _check_termination(self):
        if self.win_status is not None:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw nest
        pygame.gfxdraw.filled_circle(self.screen, int(self.nest_pos[0]), int(self.nest_pos[1]), 30, self.COLOR_NEST)
        pygame.gfxdraw.aacircle(self.screen, int(self.nest_pos[0]), int(self.nest_pos[1]), 30, self.COLOR_NEST)

        # Draw resources
        for i in range(self.N_RESOURCES):
            pos = self.resource_pos[i]
            level = self.resource_levels[i]
            color = self.COLOR_RESOURCE if level > 1 else self.COLOR_RESOURCE_DEPLETED
            
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.entity_radius + 5, color)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.entity_radius + 5, color)
            
            # Draw resource bar
            bar_width = 50
            bar_height = 8
            bar_x = pos[0] - bar_width / 2
            bar_y = pos[1] - 35
            fill_ratio = max(0, level / self.resource_max_level)
            pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_BAR_FG, (bar_x, bar_y, bar_width * fill_ratio, bar_height))
            
            # Draw resource level text
            res_text = self.font_resource.render(f"{int(level)}", True, self.COLOR_TEXT)
            self.screen.blit(res_text, (pos[0] - res_text.get_width() / 2, bar_y - 18))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p[3] / 40.0))
            if alpha > 0:
                # Create a temporary surface for the particle to handle alpha
                particle_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(particle_surf, 2, 2, 2, self.COLOR_RESOURCE + (alpha,))
                self.screen.blit(particle_surf, (int(p[0]) - 2, int(p[1]) - 2))

        # Draw ants
        for i in range(self.N_ANTS):
            pos = self.ant_positions[i]
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.entity_radius-4, self.COLOR_ANT)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.entity_radius-4, self.COLOR_ANT)

        # Draw selected ant highlight
        if self.selected_ant != -1:
            pos = self.ant_positions[self.selected_ant]
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.entity_radius, self.COLOR_SELECTED_ANT)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.entity_radius+1, self.COLOR_SELECTED_ANT)

        # Draw selector
        sel_pos = (int(self.selector_pos[0]), int(self.selector_pos[1]))
        pygame.gfxdraw.aacircle(self.screen, sel_pos[0], sel_pos[1], self.entity_radius, self.COLOR_SELECTOR)
        pygame.draw.line(self.screen, self.COLOR_SELECTOR, (sel_pos[0] - 5, sel_pos[1]), (sel_pos[0] + 5, sel_pos[1]), 1)
        pygame.draw.line(self.screen, self.COLOR_SELECTOR, (sel_pos[0], sel_pos[1] - 5), (sel_pos[0], sel_pos[1] + 5), 1)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"FOOD: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Turns
        turns_text = self.font_ui.render(f"TURNS: {self.turns} / {self.MAX_TURNS}", True, self.COLOR_TEXT)
        self.screen.blit(turns_text, (self.WIDTH - turns_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            if self.win_status == 'win':
                msg = "VICTORY"
                color = self.COLOR_WIN
            else:
                msg = "DEFEAT"
                color = self.COLOR_LOSE
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "turns": self.turns,
            "selected_ant": self.selected_ant,
            "win_status": self.win_status,
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # To play the game manually, we need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # Arrow Keys: Move selector
    # Spacebar: Select/Assign
    # Left Shift: Cycle selected ant
    # R: Reset environment
    
    print("--- Manual Control ---")
    print("Arrows: Move Selector")
    print("Space: Select Ant / Assign to Resource")
    print("Shift: Cycle Selected Ant")
    print("R: Reset")
    print("--------------------")

    # Pygame window for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Ant Colony")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0)

    # Main game loop for human play
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
        
        # Poll keyboard state
        keys = pygame.key.get_pressed()
        
        # Action 0: Movement
        mov_action = 0 # none
        if keys[pygame.K_UP]: mov_action = 1
        elif keys[pygame.K_DOWN]: mov_action = 2
        elif keys[pygame.K_LEFT]: mov_action = 3
        elif keys[pygame.K_RIGHT]: mov_action = 4
        
        # Action 1: Space
        space_action = 1 if keys[pygame.K_SPACE] else 0
        
        # Action 2: Shift
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([mov_action, space_action, shift_action])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over. Final Info: {info}")
            # Render final frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()