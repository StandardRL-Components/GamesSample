
# Generated: 2025-08-28T05:21:08.008995
# Source Brief: brief_02595.md
# Brief Index: 2595

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: ←→ to move, ↑ to jump. Reach the final exit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape a haunted mansion by jumping, dodging obstacles, and activating switches in this side-scrolling horror platformer."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME = 120  # seconds
        self.MAX_STEPS = self.MAX_TIME * self.FPS

        # Player physics
        self.GRAVITY = 0.5
        self.JUMP_STRENGTH = -10
        self.PLAYER_SPEED = 4
        self.PLAYER_FRICTION = 0.85

        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (255, 255, 255)
        self.COLOR_PLATFORM = (60, 40, 80)
        self.COLOR_PIT = (100, 20, 20)
        self.COLOR_SWITCH_OFF = (200, 200, 200)
        self.COLOR_SWITCH_ON = (100, 255, 100)
        self.COLOR_DOOR_CLOSED = (150, 120, 100)
        self.COLOR_DOOR_OPEN = (50, 40, 30)
        self.COLOR_OBSTACLE = (200, 50, 50)
        self.COLOR_TORCH_FLAME = (255, 180, 80)
        self.COLOR_TEXT = (220, 220, 220)

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # Level definitions
        self._define_levels()

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.on_ground = None
        self.particles = None
        self.current_screen_index = None
        self.level_states = None
        self.obstacle_speed_modifier = None
        self.timer = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.game_over_message = None
        
        self.reset()
        self.validate_implementation()

    def _define_levels(self):
        self.levels = [
            # Level 0: Simple jump
            {
                "player_start": [50, 300],
                "platforms": [pygame.Rect(0, 350, 250, 50), pygame.Rect(400, 350, 240, 50)],
                "pits": [pygame.Rect(250, 350, 150, 50)],
                "switches": [], "doors": [], "obstacles": [],
                "torches": [{"pos": (150, 310), "base_radius": 60}, {"pos": (500, 310), "base_radius": 60}],
                "exit_door": pygame.Rect(600, 290, 40, 60)
            },
            # Level 1: Switch and pendulum
            {
                "player_start": [50, 100],
                "platforms": [pygame.Rect(0, 150, 150, 20), pygame.Rect(200, 350, 200, 50), pygame.Rect(500, 350, 140, 50)],
                "pits": [pygame.Rect(0, 380, 640, 20)],
                "switches": [{"pos": (300, 330), "id": 0}],
                "doors": [{"rect": pygame.Rect(550, 290, 20, 60), "switch_id": 0}],
                "obstacles": [{"type": "pendulum", "pivot": (450, 50), "length": 200, "speed": 0.03, "size": 15}],
                "torches": [{"pos": (80, 110), "base_radius": 50}, {"pos": (300, 310), "base_radius": 70}],
                "exit_door": pygame.Rect(600, 290, 40, 60)
            },
            # Level 2: Sliding wall and precise jumps
            {
                "player_start": [50, 300],
                "platforms": [pygame.Rect(0, 350, 100, 50), pygame.Rect(180, 320, 80, 20), pygame.Rect(300, 280, 80, 20), pygame.Rect(450, 320, 200, 80)],
                "pits": [pygame.Rect(0, 380, 640, 20)],
                "switches": [], "doors": [],
                "obstacles": [{"type": "slider", "rect": pygame.Rect(265, 100, 30, 180), "p1": (265, 50), "p2": (265, 250), "speed": 0.05}],
                "torches": [{"pos": (60, 310), "base_radius": 50}, {"pos": (340, 240), "base_radius": 60}, {"pos": (550, 280), "base_radius": 80}],
                "exit_door": pygame.Rect(600, 260, 40, 60)
            }
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.game_over_message = ""
        self.timer = self.MAX_TIME
        self.obstacle_speed_modifier = 1.0

        self.current_screen_index = 0
        self.level_states = [{"switches_on": set()} for _ in self.levels]
        
        start_pos = self.levels[self.current_screen_index]["player_start"]
        self.player_pos = list(start_pos)
        self.player_vel = [0, 0]
        self.player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], 20, 30)
        self.on_ground = False
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        
        # --- UPDATE GAME LOGIC ---
        self.steps += 1
        self.timer = self.MAX_TIME - (self.steps / self.FPS)
        reward += 0.01  # Survival reward

        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_speed_modifier += 0.05

        self._handle_input(movement)
        self._update_player()
        
        # Check interactions and update state
        interaction_reward = self._update_interactions()
        reward += interaction_reward

        self._update_particles()
        
        # --- CHECK TERMINATION ---
        terminated = False
        if self.player_pos[1] > self.HEIGHT:
            terminated = True
            reward = -100
            self.game_over_message = "YOU FELL"
        
        current_level = self.levels[self.current_screen_index]
        for pit in current_level["pits"]:
            if self.player_rect.colliderect(pit):
                terminated = True
                reward = -100
                self.game_over_message = "YOU DIED"
                break
        
        for obstacle in self._get_current_obstacles():
            if self.player_rect.colliderect(obstacle['rect']):
                terminated = True
                reward = -100
                self.game_over_message = "YOU DIED"
                break

        if self.timer <= 0:
            terminated = True
            reward = -50
            self.game_over_message = "TIME'S UP"
        
        if self.game_won:
            terminated = True
            reward = 100
            self.game_over_message = "YOU ESCAPED!"

        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel[0] = self.PLAYER_SPEED
        else:
            self.player_vel[0] *= self.PLAYER_FRICTION

        # Vertical movement (Jump)
        if movement == 1 and self.on_ground:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            # sound: player_jump.wav
            for _ in range(10):
                self.particles.append({
                    "pos": [self.player_rect.centerx, self.player_rect.bottom],
                    "vel": [random.uniform(-1, 1), random.uniform(0, 1)],
                    "life": random.randint(10, 20),
                    "radius": random.uniform(1, 3),
                    "color": (100, 80, 120)
                })

    def _update_player(self):
        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        
        # Update position
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]

        # Keep player within horizontal bounds
        self.player_pos[0] = max(0, min(self.WIDTH - self.player_rect.width, self.player_pos[0]))

        # Update rect and check collisions
        self.player_rect.x = int(self.player_pos[0])
        self.player_rect.y = int(self.player_pos[1])

        self.on_ground = False
        current_level = self.levels[self.current_screen_index]
        for platform in current_level["platforms"]:
            if self.player_rect.colliderect(platform):
                if self.player_vel[1] > 0 and self.player_rect.bottom < platform.top + self.player_vel[1] + 1:
                    self.player_rect.bottom = platform.top
                    self.player_pos[1] = self.player_rect.y
                    if not self.on_ground: # Landing
                         # sound: player_land.wav
                         for _ in range(5):
                            self.particles.append({
                                "pos": [self.player_rect.centerx, self.player_rect.bottom],
                                "vel": [random.uniform(-1.5, 1.5), random.uniform(-0.5, 0)],
                                "life": random.randint(5, 15), "radius": random.uniform(1, 2), "color": (100, 80, 120)
                            })
                    self.player_vel[1] = 0
                    self.on_ground = True
                elif self.player_vel[1] < 0 and self.player_rect.top > platform.bottom + self.player_vel[1] -1:
                    self.player_rect.top = platform.bottom
                    self.player_pos[1] = self.player_rect.y
                    self.player_vel[1] = 0 # Bonk head

    def _update_interactions(self):
        reward = 0
        current_level_idx = self.current_screen_index
        current_level = self.levels[current_level_idx]
        current_level_state = self.level_states[current_level_idx]

        # Check switch interactions
        for switch in current_level["switches"]:
            switch_rect = pygame.Rect(switch["pos"][0] - 5, switch["pos"][1] - 5, 10, 10)
            if self.player_rect.colliderect(switch_rect):
                if switch["id"] not in current_level_state["switches_on"]:
                    current_level_state["switches_on"].add(switch["id"])
                    reward += 5
                    # sound: switch_activate.wav
                    for _ in range(20):
                        self.particles.append({
                            "pos": list(switch["pos"]),
                            "vel": [random.uniform(-2, 2), random.uniform(-3, 1)],
                            "life": random.randint(15, 30), "radius": random.uniform(1, 4), "color": self.COLOR_SWITCH_ON
                        })
        
        # Check exit door
        if self.player_rect.colliderect(current_level["exit_door"]):
            is_door_open = True
            for door in current_level["doors"]:
                if door["switch_id"] not in current_level_state["switches_on"]:
                    is_door_open = False
                    break
            
            if is_door_open:
                if self.current_screen_index < len(self.levels) - 1:
                    self.current_screen_index += 1
                    start_pos = self.levels[self.current_screen_index]["player_start"]
                    self.player_pos = list(start_pos)
                    self.player_vel = [0, 0]
                    reward += 10
                    # sound: next_screen.wav
                else:
                    self.game_won = True
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_current_obstacles(self):
        # Calculate current positions of dynamic obstacles for collision and rendering
        dynamic_obstacles = []
        current_level = self.levels[self.current_screen_index]
        for obs in current_level["obstacles"]:
            if obs['type'] == 'pendulum':
                angle = math.pi/2 + math.sin(self.steps * obs['speed'] * self.obstacle_speed_modifier) * (math.pi/3)
                end_x = obs['pivot'][0] + obs['length'] * math.cos(angle)
                end_y = obs['pivot'][1] + obs['length'] * math.sin(angle)
                dynamic_obstacles.append({
                    'type': 'pendulum', 'pivot': obs['pivot'], 'pos': (end_x, end_y),
                    'rect': pygame.Rect(end_x - obs['size'], end_y - obs['size'], obs['size']*2, obs['size']*2),
                    'size': obs['size']
                })
            elif obs['type'] == 'slider':
                progress = (math.sin(self.steps * obs['speed'] * self.obstacle_speed_modifier) + 1) / 2
                curr_x = obs['p1'][0] + (obs['p2'][0] - obs['p1'][0]) * progress
                curr_y = obs['p1'][1] + (obs['p2'][1] - obs['p1'][1]) * progress
                rect = obs['rect'].copy()
                rect.topleft = (curr_x, curr_y)
                dynamic_obstacles.append({'type': 'slider', 'rect': rect})
        return dynamic_obstacles

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background_effects()
        self._render_level()
        self._render_obstacles()
        self._render_player()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_effects(self):
        current_level = self.levels[self.current_screen_index]
        for torch in current_level["torches"]:
            light_radius = torch['base_radius'] + random.randint(-5, 5)
            light_alpha = random.randint(30, 50)
            s = pygame.Surface((light_radius * 2, light_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_TORCH_FLAME + (light_alpha,), (light_radius, light_radius), light_radius)
            self.screen.blit(s, (torch['pos'][0] - light_radius, torch['pos'][1] - light_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Flame particle
            flame_color = (random.randint(200,255), random.randint(100,200), 0)
            flame_pos = (torch['pos'][0] + random.randint(-2, 2), torch['pos'][1] - 5 + random.randint(-3, 3))
            pygame.gfxdraw.filled_circle(self.screen, int(flame_pos[0]), int(flame_pos[1]), random.randint(2,4), flame_color)


    def _render_level(self):
        current_level_idx = self.current_screen_index
        current_level = self.levels[current_level_idx]
        current_level_state = self.level_states[current_level_idx]

        for platform in current_level["platforms"]:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, platform)
        for pit in current_level["pits"]:
            pygame.draw.rect(self.screen, self.COLOR_PIT, pit)
        
        for switch in current_level["switches"]:
            color = self.COLOR_SWITCH_ON if switch["id"] in current_level_state["switches_on"] else self.COLOR_SWITCH_OFF
            pos = (int(switch["pos"][0]), int(switch["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, color)

        for door in current_level["doors"]:
            color = self.COLOR_DOOR_OPEN if door["switch_id"] in current_level_state["switches_on"] else self.COLOR_DOOR_CLOSED
            pygame.draw.rect(self.screen, color, door["rect"])

        pygame.draw.rect(self.screen, self.COLOR_DOOR_OPEN, current_level["exit_door"])

    def _render_obstacles(self):
        for obs in self._get_current_obstacles():
            if obs['type'] == 'pendulum':
                pygame.draw.aaline(self.screen, (50,50,50), obs['pivot'], obs['pos'])
                pygame.gfxdraw.filled_circle(self.screen, int(obs['pos'][0]), int(obs['pos'][1]), obs['size'], self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, int(obs['pos'][0]), int(obs['pos'][1]), obs['size'], self.COLOR_OBSTACLE)
            elif obs['type'] == 'slider':
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])

    def _render_player(self):
        # Glow effect
        glow_rect = self.player_rect.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_PLAYER_GLOW + (30,), s.get_rect(), border_radius=5)
        self.screen.blit(s, glow_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

    def _render_ui(self):
        # Timer
        timer_text = f"TIME: {max(0, int(self.timer))}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            game_over_surf = self.font_game_over.render(self.game_over_message, True, self.COLOR_TEXT)
            pos = (self.WIDTH // 2 - game_over_surf.get_width() // 2, self.HEIGHT // 2 - game_over_surf.get_height() // 2)
            self.screen.blit(game_over_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "current_screen": self.current_screen_index,
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        import os
        # Set a non-dummy video driver for human rendering
        if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
            del os.environ["SDL_VIDEODRIVER"]
            
    env = GameEnv(render_mode="rgb_array")

    if render_mode == "human":
        # Pygame setup for human rendering
        pygame.init()
        human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Haunted Mansion Escape")
        clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    # --- Human Controls ---
    # This maps keyboard keys to the MultiDiscrete action space
    # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    # actions[1]: Space button (0=released, 1=held) -> Unused
    # actions[2]: Shift button (0=released, 1=held) -> Unused
    action = np.array([0, 0, 0]) 

    while not done:
        if render_mode == "human":
            # --- Human Input Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            
            # Reset action
            action[0] = 0 # No movement
            
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2 # No-op in this game
            
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # --- Step the environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- Render to the screen ---
            # The observation is already a rendered frame
            # We just need to convert it back to a Pygame surface
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(env.FPS)
        
        else: # For RL agent
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
                obs, info = env.reset()

    env.close()